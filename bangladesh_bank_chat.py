import os
import re
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bangladesh-bank-docs")

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_query(query):
    """Create embedding for the search query"""
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return None


def search_knowledge_base(query, top_k=30, alpha=0.5):
    """Search for relevant chunks in the knowledge base"""
    query_embedding = embed_query(query)

    if not query_embedding:
        return []

    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            alpha=alpha
        )

        relevant_chunks = []
        for match in results.matches:
            if match.score > 0.3:  # Lower threshold for better recall
                relevant_chunks.append({
                    'text': match.metadata.get('text', ''),
                    'source': match.metadata.get('source', 'Unknown'),
                    'score': match.score
                })

        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        return relevant_chunks

    except Exception as e:
        print(f"Search error: {e}")
        return []


def group_chunks_by_context(chunks, max_context_length=3000):
    """Group chunks by source to maintain context"""
    if not chunks:
        return []

    # Group by source
    source_groups = {}
    for chunk in chunks:
        key = chunk['source']
        if key not in source_groups:
            source_groups[key] = []
        source_groups[key].append(chunk)

    # Create context groups
    context_groups = []
    current_group = []
    current_length = 0

    # Sort sources by highest relevance score
    sorted_sources = sorted(
        source_groups.keys(),
        key=lambda k: max(chunk['score'] for chunk in source_groups[k]),
        reverse=True
    )

    for source_key in sorted_sources:
        source_chunks = source_groups[source_key]
        source_text = " ".join([chunk['text'] for chunk in source_chunks])

        if current_length + len(source_text) <= max_context_length:
            current_group.extend(source_chunks)
            current_length += len(source_text)
        else:
            if current_group:
                context_groups.append(current_group)
            current_group = source_chunks[:]
            current_length = len(source_text)

    if current_group:
        context_groups.append(current_group)

    return context_groups


def format_conversation_history(conversation_history, max_turns=5):
    """Format conversation history for inclusion in the prompt"""
    if not conversation_history:
        return ""
    
    # Take only the last N turns to avoid token overflow
    recent_history = conversation_history[-max_turns:]
    
    formatted = "Recent conversation:\n"
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    
    return formatted


def answer_question_with_context(query, context_groups, language,conversation_history=None):
    """Generate answer using OpenAI with provided context"""
    # Removed the early return for empty context_groups
    # Now always proceed to build prompt, even if contexts are empty

    all_contexts = []
    sources_info = []

    for i, group in enumerate(context_groups):
        group_text = ""
        group_sources = set()

        for chunk in group:
            group_text += chunk['text'] + " "
            group_sources.add(chunk['source'])

        context_header = f"Context {i+1} (Source: {', '.join(group_sources)}):\n"
        all_contexts.append(context_header + group_text.strip())
        sources_info.extend([(chunk['source'], chunk['score']) for chunk in group])

    full_context = "\n\n".join(all_contexts) if all_contexts else "No relevant contexts found."

    # Format conversation history
    formatted_history = format_conversation_history(conversation_history)

    # Language-specific prompts
    if language == "bangla":
        prompt =  f"""আপনি 'বাংলাদেশ ব্যাংক'-এর বিধি ও প্রবিধানে বিশেষজ্ঞ একজন পরামর্শদাতা। আপনার ভূমিকা হলো ব্যাংকিং-সম্পর্কিত বিষয়ে স্পষ্ট, ব্যবহারিক এবং নির্ভুল দিকনির্দেশনা প্রদান করা। 'বাংলাদেশ ব্যাংক'-এর ব্যাংকিং প্রবিধানের আওতার মধ্যেই থাকুন।

        কথোপকথনের ইতিহাস:
        {formatted_history}

        নির্দেশনা:
        1. সমস্ত প্রদত্ত প্রেক্ষাপট (কনটেক্সট) মনোযোগসহ বিশ্লেষণ করুন এবং নিয়ম ও প্রেক্ষাপট বুঝে সঠিক ও সরাসরি উত্তর প্রদান করুন (অনুমান করবেন না)
        2. বিস্তারিত এবং সম্পূর্ণ উত্তর প্রদান করুন - প্রাসঙ্গিক সকল নিয়ম, বিধি এবং পদ্ধতিসমূহ ব্যাখ্যা করুন
        3. প্রাসঙ্গিক হলে একাধিক প্রসঙ্গ থেকে তথ্য ব্যবহার করুন এবং সকল প্রাসঙ্গিক বিধিমালা অন্তর্ভুক্ত করুন

        **উদ্ধৃতি নিয়ম (অত্যন্ত গুরুত্বপূর্ণ):**
        4. প্রতিটি গুরুত্বপূর্ণ তথ্য, নিয়ম, বিধি বা পদক্ষেপের জন্য অবশ্যই সূত্র উল্লেখ করুন
        5. সূত্র উল্লেখের ফরম্যাট: [নিয়ম/বিধি/সার্কুলার নং, প্রকাশের তারিখ]
        6. **উদাহরণ দেখুন এবং অবশ্যই অনুসরণ করুন:**
        - "ব্যাংকসমূহকে ঋণ শ্রেণীকরণ নির্দেশিকা মেনে চলতে হবে [BRPD সার্কুলার নং ১৪, তারিখ: ২৩ সেপ্টেম্বর ২০১২]"
        - "মূলধন পর্যাপ্ততার অনুপাত ন্যূনতম ১০% হতে হবে [BRPD সার্কুলার নং ০১, জানুয়ারি ২০২০]"
        - "সন্দেহজনক লেনদেন রিপোর্ট করা বাধ্যতামূলক [মানিলন্ডারিং প্রতিরোধ আইন ২০১২, ধারা ২৩(১)]"
        - "গ্রাহক সনাক্তকরণ পদ্ধতি সম্পন্ন করতে হবে [নো ইউর কাস্টমার (KYC) নির্দেশিকা, BRPD সার্কুলার নং ০৫]"
        - "একাধিক সার্কুলার থেকে তথ্য থাকলে [BRPD সার্কুলার নং ১৬, অক্টোবর ৩০, ২০১৮; BRPD সার্কুলার লেটার নং ১২, জুন ২০, ২০১৯]"
        
        সূত্রের নম্বর পরিবর্তন করবেন না যেমন ২ থেকে খ ইত্যাদি।

        **গুরুত্বপূর্ণ উদ্ধৃতি নির্দেশনা:**
        7. Context থেকে সার্কুলার নম্বর, নিয়ম নম্বর, এবং তারিখ সঠিকভাবে তুলে ধরুন
        8. প্রতিটি গুরুত্বপূর্ণ বিবৃতিতে অন্তত একটি সূত্র উল্লেখ করুন
        9. PDF নাম এবং পৃষ্ঠা নম্বর উল্লেখ করবেন না - শুধুমাত্র নিয়ম/সার্কুলার নম্বর এবং তারিখ ব্যবহার করুন
        10. একাধিক সার্কুলার থেকে তথ্য থাকলে সেমিকোলন (;) দিয়ে আলাদা করুন
        11. প্রদত্ত প্রসঙ্গের তথ্যের উপর সম্পূর্ণভাবে ভিত্তি করে উত্তর দিন - প্রসঙ্গে পর্যাপ্ত তথ্য না থাকলে স্পষ্টভাবে বলুন

        **ফরম্যাটিং নির্দেশনা:**
        11. সঠিক বাংলা মার্কডাউন ফরম্যাটে উত্তর দিন
        12. প্রয়োজনে নম্বরযুক্ত তালিকা, বুলেট পয়েন্ট এবং **বোল্ড টেক্সট** ব্যবহার করুন
        13. চূড়ান্ত উত্তরে "প্রসঙ্গ নম্বর" বা "কনটেক্সট গোষ্ঠী" শব্দগুলি উল্লেখ করবেন না
        14. সবসময় বাংলায় উত্তর দিন
        15. পূর্ববর্তী কথোপকথন বিবেচনা করুন যদি এটি বর্তমান প্রশ্নের সাথে প্রাসঙ্গিক হয়
        16. একজন পেশাদার পরামর্শদাতা হিসেবে শুভেচ্ছা এবং সাধারণ প্রশ্নের উত্তর দিন, যিনি সহায়ক, বিনয়ী এবং জ্ঞানী

        নথি থেকে প্রসঙ্গ:
        {full_context}

        প্রশ্ন: {query}

        **গুরুত্বপূর্ণ সতর্কতা: 
        - আপনার উত্তর অবশ্যই বাংলায় হতে হবে
        - প্রতিটি তথ্যের জন্য [নিয়ম/সার্কুলার নং, তারিখ] ফরম্যাটে সূত্র উল্লেখ করতে হবে
        - PDF নাম এবং পৃষ্ঠা নম্বর উল্লেখ করবেন না
        - Context থেকে সঠিক সার্কুলার নম্বর এবং তারিখ ব্যবহার করুন
        - একাধিক সার্কুলার থাকলে সেমিকোলন দিয়ে আলাদা করুন যেমন: [BRPD সার্কুলার নং ১৬, অক্টোবর ৩০, ২০১৮; BRPD সার্কুলার লেটার নং ১২, জুন ২০, ২০১৯]**

        উত্তর:"""
    else:
        prompt = f"""You are an expert consultant specializing in "Bangladesh Bank" rules and regulations. Your role is to provide clear, practical, and accurate guidance on banking-related matters. Stay within the scope of banking regulations in "Bangladesh Bank".

        Conversation History:
        {formatted_history}

        INSTRUCTIONS:
        1. Analyze all the provided contexts carefully and provide accurate, direct answers based on understanding the rules and context (don't speculate)
        2. Provide a comprehensive and complete answer - explain all relevant rules, regulations, and procedures
        3. Use information from multiple contexts when relevant and include all applicable regulations

        **CITATION REQUIREMENTS (CRITICAL - MUST FOLLOW):**
        4. You MUST cite sources for every important fact, rule, regulation, or procedure
        5. Citation format: [Rule/Circular/Regulation number, Date published]
        6. **Follow these examples and use them as a template:**
        - "Banks must comply with loan classification guidelines [BRPD Circular No. 14, dated September 23, 2012]"
        - "The minimum capital adequacy ratio must be 10% [BRPD Circular No. 01, January 2020]"
        - "Reporting suspicious transactions is mandatory [Money Laundering Prevention Act 2012, Section 23(1)]"
        - "Customer identification procedures must be completed [Know Your Customer (KYC) Guidelines, BRPD Circular No. 05]"
        - "When multiple circulars are referenced [BRPD Circular No. 16, October 30, 2018; BRPD Circular Letter No. 12, June 20, 2019]"
        
        Don't change citation numbers like 2 to খ etc.

        **IMPORTANT CITATION GUIDELINES:**
        7. Extract circular numbers, rule numbers, and dates accurately from the context
        8. Each major statement should have at least one citation
        9. Do NOT mention PDF names or page numbers - only use rule/circular numbers and dates
        10. When multiple circulars are referenced, separate them with semicolons (;)
        11. Base your answer entirely on the given contexts - if contexts don't contain enough information, state this clearly

        **FORMATTING GUIDELINES:**
        11. Answer in proper markdown format
        12. Use numbered lists, bullet points, and **bold text** where appropriate
        13. Do NOT mention "context numbers" or "context groups" in the final answer
        14. Always respond in English
        15. Consider previous conversations if they are relevant to the current question
        16. Greet and answer general questions as a professional consultant who is helpful, polite, and knowledgeable

        CONTEXTS FROM DOCUMENT:
        {full_context}

        QUESTION: {query}

        **CRITICAL REMINDERS:
        - Your answer MUST be in English
        - MUST include citations in [Rule/Circular number, Date] format for every piece of information
        - Do NOT mention PDF names or page numbers - only circular/rule numbers and dates
        - Use correct circular numbers and dates from the context
        - When multiple circulars are referenced, separate with semicolons like: [BRPD Circular No. 16, October 30, 2018; BRPD Circular Letter No. 12, June 20, 2019]**

        ANSWER:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content, sources_info
    except Exception as e:
        print(f"Error generating answer: {e}")
        if language == "bangla":
            return f"দুঃখিত, উত্তর তৈরিতে সমস্যা হয়েছে: {str(e)}", []
        return f"Sorry, having some issue with generating answer: {str(e)}", []


def get_answer(query, language, conversation_history=None):
    """Main function to get answer for a query"""
    
    print(f"Using language: {language}")

    # Search for relevant chunks
    relevant_chunks = search_knowledge_base(query, top_k=15, alpha=0.5)

    print(f"Found {len(relevant_chunks)} relevant chunks")

    # Group chunks by context (even if empty)
    context_groups = group_chunks_by_context(relevant_chunks, max_context_length=3000)

    context_groups = context_groups[:7]
    print(f"Grouped into {len(context_groups)} context groups")

    # Always generate answer via LLM, even with empty context
    answer, sources = answer_question_with_context(query, context_groups, language, conversation_history)

    return answer, sources


def get_consultation(text_input, language, conversation_history=None):
    """Wrapper function for easy use"""
    start_time = time.perf_counter()
    answer, sources = get_answer(text_input, language, conversation_history)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return {
        "answer": answer,
        "language": language,
        "TotalTime": f"{total_time:.2f} seconds"
    }


# Example Usage:
if __name__ == "__main__":

    result = get_consultation(
        text_input="tell me what is the Guidelines on Internal  Credit  Risk  Rating  System  for  Banks",
        language="english",
        conversation_history=[
            {"role": "user", "content": "What is the capital requirement for banks in Bangladesh?"},
            {"role": "assistant", "content": "The capital requirement for banks in Bangladesh is set by the Bangladesh Bank and varies based on the type of bank."}
        ]
    )
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nLanguage:\n{result['language']}")
    print(f"\nTime:\n{result['TotalTime']}")