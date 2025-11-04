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


def search_knowledge_base(query, top_k=10, alpha=0.5):
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


def group_chunks_by_context(chunks, max_context_length=2500):
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


def answer_question_with_context(query, context_groups, language):
    """Generate answer using OpenAI with provided context"""
    if not context_groups:
        if language == "bangla":
            return "দুঃখিত, প্রশ্নের সাথে সম্পর্কিত পর্যাপ্ত তথ্য পাওয়া যায়নি।", []
        return "Not enough relevant information found.", []

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

    full_context = "\n\n".join(all_contexts)

    # Language-specific prompts
    if language == "bangla":
        prompt = f"""আপনি বাংলাদেশের ব্যাংক নিয়ম ও প্রবিধান বিষয়ে একজন বিশেষজ্ঞ পরামর্শদাতা। আপনার দায়িত্ব হলো ব্যাংকিং সংক্রান্ত বিষয়ে স্পষ্ট, ব্যবহারিক এবং নির্ভুল পরামর্শ প্রদান করা।

        নির্দেশনা:
        1. সমস্ত প্রদত্ত প্রেক্ষাপট (কনটেক্সট) মনোযোগসহ বিশ্লেষণ করুন এবং নিয়ম ও প্রেক্ষাপট বুঝে সর্বপ্রথম সঠিক, সংক্ষিপ্ত ও সরাসরি উত্তর বা সমাধান দিন; প্রয়োজনে পরে ব্যাখ্যা করুন। (অনুমান করবেন না)
        2. প্রাসঙ্গিক হলে একাধিক প্রসঙ্গ থেকে তথ্য ব্যবহার করুন
        3. শুধুমাত্র প্রদত্ত প্রসঙ্গের উপর ভিত্তি করে একটি বিস্তৃত উত্তর প্রদান করুন, এবং উত্সগুলির রেফারেন্স প্রদান করুন (সম্ভব হলে নিয়মের নম্বর, বিভাগ নম্বর, প্রকাশের তারিখ ইত্যাদি প্রদান করুন)
        4. প্রসঙ্গ এবং এটি কী বোঝায় তা বুঝুন এবং সর্বোত্তম সম্ভাব্য উত্তর প্রদান করুন
        5. প্রসঙ্গে পর্যাপ্ত তথ্য না থাকলে, এটি স্পষ্টভাবে বলুন
        6. আপনার প্রতিক্রিয়ায় সুনির্দিষ্ট এবং বিস্তারিত হন
        7. সবসময় বাংলায় উত্তর দিন
        8. সঠিক মার্কডাউন ফরম্যাটে উত্তর দিন

        নথি থেকে প্রসঙ্গ:
        {full_context}

        প্রশ্ন: {query}

        **IMPORTANT: আপনার উত্তর অবশ্যই বাংলায় হতে হবে।**

        উত্তর:"""
    else:
        prompt = f"""You are an expert consultant specializing in Bangladesh banks' rules and regulations. Your role is to provide clear, practical, and accurate guidance on banking-related matters.

        INSTRUCTIONS:
        1. Analyze all the provided contexts carefully and give the accurate short direct answer or solution first by understanding the rules and context then explain if needed. (don't speculate)
        2. Use information from multiple contexts when relevant
        3. Provide a comprehensive answer based only on the given contexts, and provide references to the sources (Porvide rules no. section no. date published etc. if possible)
        4. Understand the context and what it implies to provide the best possible answer
        5. If contexts don't contain enough information, say so clearly
        6. Be specific and detailed in your response
        7. Always respond in English
        8. Answer in proper markdown format

        CONTEXTS FROM DOCUMENT:
        {full_context}

        QUESTION: {query}

        **IMPORTANT: Your answer MUST be in English.**

        ANSWER:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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


def get_answer(query, language):
    """Main function to get answer for a query"""
    
    print(f"Using language: {language}")

    # Search for relevant chunks
    relevant_chunks = search_knowledge_base(query, top_k=10)

    if not relevant_chunks:
        if language == "bangla":
            return "দুঃখিত, প্রশ্নের সাথে সম্পর্কিত কোনো তথ্য পাওয়া যায়নি।", []
        return "No relevant answer found for the given question.", []

    print(f"Found {len(relevant_chunks)} relevant chunks")

    # Group chunks by context
    context_groups = group_chunks_by_context(relevant_chunks, max_context_length=2500)

    context_groups = context_groups[:5]
    print(f"Grouped into {len(context_groups)} context groups")

    # Generate answer
    answer, sources = answer_question_with_context(query, context_groups, language)

    return answer, sources


def get_consultation(text_input, language):
    """Wrapper function for easy use"""
    start_time = time.perf_counter()
    answer, sources = get_answer(text_input, language)
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
        text_input="What regulation for Bank Resolution?",
        language="english"
    )
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nLanguage:\n{result['language']}")
    print(f"\nTime:\n{result['TotalTime']}")