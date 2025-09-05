# Sales Agent Feature

## Overview
The chatbot has been updated to act as an expert sales agent that suggests competitor alternatives for any product inquiry. This feature helps customers make informed decisions by presenting multiple product options with positive points about each competitor.

## Key Features

### 1. Universal Competitor Suggestions
- **Any Product**: The system now suggests competitors for ANY product the user asks about, not just specific ones
- **Positive Focus**: Highlights positive points and advantages of competitor products
- **Informed Decision Making**: Helps customers understand their options before making a purchase

### 2. Implementation Details

#### Modified Components:
1. **Web Service (`app/core/web_service.py`)**
   - Updated system prompts to act as sales agent
   - Added competitor suggestion instructions for both main and fallback responses
   - Ensures 2-3 competitor alternatives are always suggested

2. **RAG Service (`app/core/rag_service.py`)**
   - Enhanced document-based responses to include competitor suggestions
   - Maintains sales-focused approach even with document evidence

3. **Orchestrator (`app/core/orchestrator.py`)**
   - Updated fusion logic to preserve sales agent behavior when combining RAG and web results
   - Ensures consistent competitor suggestions across all response types

#### Key Prompt Changes:
- **Primary System Prompt**: "You are an expert sales agent who synthesizes accurate, sales-ready answers from multiple web sources with inline [n] citations. Always suggest competitor alternatives with their positive benefits and advantages. Act in the customer's best interest by presenting multiple product options and helping them make informed decisions."

- **Response Structure**: The system now follows a specific structure:
  1. Concise answer about the requested product first
  2. Bullets with value props/specs of the requested product  
  3. **LAST**: 'Competitor Alternatives' section with 2-3 alternatives and their positive points

- **Competitor Instructions**: "Structure your response: Answer about requested product first, then LAST: Add 'Competitor Alternatives' section with positive points about alternatives to help customers make informed decisions."

### 3. Examples of Expected Behavior

#### User asks about iPhone 16:
- Provides iPhone 16 information and features first
- Lists iPhone 16 specifications and value propositions
- **At the end**: Adds 'Competitor Alternatives' section suggesting Samsung Galaxy S24/S25 series with positive points (better customization, S Pen, camera zoom, faster charging) and other alternatives like Google Pixel or OnePlus

#### User asks about Tesla Model 3:
- Provides Tesla Model 3 information  
- Suggests competitors like BMW i4, Mercedes EQA, Polestar 2 with their advantages
- Highlights unique selling points of each alternative

#### User asks about MacBook Pro:
- Provides MacBook Pro information
- Suggests Windows alternatives like Dell XPS, Surface Laptop, ThinkPad X1 with their benefits
- Mentions advantages like better gaming, more ports, upgradeability

### 4. Technical Implementation

The sales agent behavior is implemented at multiple layers:

1. **Web Search Layer**: When searching for information, the system prompts are configured to always look for and suggest competitors

2. **RAG Layer**: When using document-based knowledge, the synthesis includes competitor suggestions based on the knowledge base

3. **Fusion Layer**: When combining web and document information, the final response maintains the sales agent perspective

### 5. Benefits

- **Customer-Centric**: Helps customers make informed decisions
- **Comprehensive**: Covers all product categories automatically
- **Competitive Analysis**: Provides market awareness for better choices
- **Sales Excellence**: Positions the bot as a knowledgeable sales consultant

### 6. Usage

The feature works automatically with any product query. Simply ask about any product, and the system will:

1. Answer the specific question about the requested product
2. Suggest 2-3 relevant competitor alternatives
3. Highlight positive aspects of each competitor
4. Help the user understand their options

### 7. Configuration

The sales agent behavior is built into the core prompts and doesn't require additional configuration. It works with:

- Web search results (when available)
- Document-based knowledge (RAG)
- Fallback responses (when no sources are available)
- All product categories and industries

This makes the chatbot a comprehensive sales consultant that always acts in the customer's best interest by providing multiple options and helping them make informed purchasing decisions.
