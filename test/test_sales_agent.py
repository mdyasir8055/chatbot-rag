#!/usr/bin/env python3
"""
Test script to verify the sales agent behavior
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.orchestrator import Orchestrator

async def test_sales_agent():
    """Test the sales agent behavior with different products"""
    
    orch = Orchestrator()
    
    # Test cases with different products
    test_cases = [
        "Tell me about iPhone 16",
        "What are the features of Samsung Galaxy S24?",
        "I want to buy a Tesla Model 3",
        "What about MacBook Pro M3?",
        "Tell me about PlayStation 5",
        "I'm interested in Nike Air Max shoes"
    ]
    
    print("🤖 Testing Sales Agent Behavior\n" + "="*50)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n📱 Test {i}: {question}")
        print("-" * 40)
        
        try:
            # Initialize with a simple product name to set context
            product_name = question.split()[-1] if question.split() else "Product"
            await orch.init_topic(product_name=product_name, rag_enabled=False)
            
            # Ask the question
            result = await orch.answer_dual(question)
            
            # Display the result
            print(f"🤖 Response: {result.get('final_answer', 'No response')}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_sales_agent())
