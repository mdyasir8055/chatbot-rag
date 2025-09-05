#!/usr/bin/env python3
"""
Test script to verify competitor alternatives appear at the end of responses
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.orchestrator import Orchestrator

async def test_competitor_order():
    """Test that competitors appear at the end of responses"""
    
    orch = Orchestrator()
    
    print("🤖 Testing Competitor Order (Should appear at END)\n" + "="*60)
    
    # Test with a simple product query
    question = "Tell me about iPhone 16 battery life"
    
    try:
        # Initialize with product name
        await orch.init_topic(product_name="iPhone 16", rag_enabled=False)
        
        # Ask the question
        result = await orch.answer_dual(question)
        
        # Display the result
        response = result.get('final_answer', 'No response')
        print(f"📱 Question: {question}")
        print("-" * 40)
        print(f"🤖 Response:\n{response}")
        print("-" * 40)
        
        # Check if "Competitor Alternatives" appears near the end
        if "competitor" in response.lower():
            lines = response.split('\n')
            competitor_line = -1
            for i, line in enumerate(lines):
                if "competitor" in line.lower():
                    competitor_line = i
                    break
            
            if competitor_line > len(lines) * 0.6:  # Should be in last 40% of response
                print("✅ SUCCESS: Competitors appear towards the end of response")
            else:
                print("❌ ISSUE: Competitors appear too early in response")
        else:
            print("❌ ISSUE: No competitor alternatives found in response")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_competitor_order())
