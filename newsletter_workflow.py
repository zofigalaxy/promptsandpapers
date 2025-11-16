#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated ArXiv Newsletter Agent
Sends personalized research digests and learns from user feedback
"""

import json
import os
from datetime import datetime, timedelta
from openai import OpenAI
from scraper_functions import ArxivWebScraper
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from supabase import create_client, Client
import urllib.parse
import hashlib
import re
import random
import time
import sys

# ============================================
# CONFIGURATION
# ============================================

OPENAI_KEY = os.environ.get('OPENAI_KEY', '')
SENDGRID_KEY = os.environ.get('SENDGRID_KEY', '')
FROM_EMAIL = os.environ.get('FROM_EMAIL', '')
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
SEND_EMPTY_DIGESTS = True

# Constants
SEND_EMPTY_DIGESTS = True
MIN_VOTES_FOR_ANALYSIS = 20
MIN_RELEVANT_VOTES = 5
MIN_NOT_RELEVANT_VOTES = 3
ANALYSIS_COOLDOWN_DAYS = 14
CONFIDENCE_THRESHOLD = 0.75
MAX_PAPERS_FOR_ANALYSIS = 20
ANALYSIS_COOLDOWN_DAYS = 14
ANALYSIS_COOLDOWN_ON_FAILURE = 1
MAX_EMAIL_RETRIES = 2

# Pdf settings
READ_FULL_PDFS = True  # If True, download and read full PDFs for detailed reviews
MAX_AUTHORS_DISPLAY = 10

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# ============================================
# DATABASE FUNCTIONS
# ============================================

def load_subscribers():
    """Load all active subscribers from Supabase"""
    try:
        print("Querying Supabase for active subscribers...")
        response = supabase.table('user_profiles').select("*").eq('active', True).eq('email_enabled', True).execute()
        return response.data
        
    except Exception as e:
        print(f"Error loading subscribers: {e}")
        return []

def update_last_sent(email, timestamp):
    """Update last_sent timestamp in Supabase"""
    try:
        supabase.table('user_profiles').update({
            'last_sent': timestamp
        }).eq('email', email).execute()
        return True
        
    except Exception as e:
        print(f"Error updating timestamp: {e}")
        return False

def store_paper(user_id, paper):
    """
    Store a paper in the sent_papers table.
    
    Args:
        user_id: User's UUID
        paper: Paper dictionary with arxiv_id, title, authors, etc.
    
    Returns:
        True if stored successfully, False if already exists or error
    """
    try:
        # Check if already exists
        existing = supabase.table('sent_papers')\
            .select('id')\
            .eq('user_id', user_id)\
            .eq('arxiv_id', paper['arxiv_id'])\
            .execute()

        #Paper already exists
        if existing.data:
            return False
        
        # Insert new paper
        supabase.table('sent_papers').insert({
            'user_id': user_id,
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'abstract': paper.get('abstract', ''),
            'review': paper['review'],
            'arxiv_link': paper['arxiv_link'],
            'pdf_link': paper['pdf_link'],
            'processed_at': datetime.now().isoformat()
        }).execute()
        
        return True
    except Exception as e:
        print(f"Error storing paper {paper['arxiv_id']}: {e}")
        return False

# ============================================
# PROMPT EVOLUTION BASED ON USERS' VOTES
# ============================================

def _format_papers_for_analysis(papers, limit):
    """
    Format papers for GPT analysis with title and abstract snippets.
    
    Args:
        papers: List of paper dictionaries
        limit: Maximum papers to include
    
    Returns:
        Formatted string
    """
    formatted = ""
    
    for i, paper in enumerate(papers[:limit]):
        formatted += f"\n{i+1}. Title: {paper['paper_title']}\n"
        
        if paper.get('paper_abstract'):
            abstract_snippet = paper['paper_abstract'][:600]
            if len(paper['paper_abstract']) > 600:
                abstract_snippet += "..."
            formatted += f"   Abstract: {abstract_snippet}\n"
    
    if len(papers) > limit:
        formatted += f"\n... and {len(papers) - limit} more papers\n"
    
    return formatted

def _build_pattern_analysis_prompt(relevant_papers, irrelevant_papers, relevant_formatted, irrelevant_formatted):
    """
    Build GPT-4 prompt for analyzing voting patterns.
    
    Returns:
        Complete prompt string
    """
    return f"""You are analyzing a researcher's voting patterns to improve their paper classification system.

The researcher has voted on {len(relevant_papers) + len(irrelevant_papers)} papers:
- {len(relevant_papers)} marked as RELEVANT
- {len(irrelevant_papers)} marked as NOT RELEVANT

Here are the papers they marked RELEVANT:
{relevant_formatted}

Here are the papers they marked NOT RELEVANT:
{irrelevant_formatted}

Your task is to identify clear, actionable patterns that could improve their classification prompt.

Analyze these votes and identify:

1. STRONG POSITIVE PATTERNS
   - Topics, methods, instruments, or research areas that appear FREQUENTLY in RELEVANT papers but RARELY/NEVER in IRRELEVANT papers
   - Look for: specific instruments (JWST, Gaia), methods (machine learning, photometry), objects (dwarf galaxies, LSB features), research areas
   - Example: "User consistently marks papers using JWST near-infrared data as relevant (e.g. 9/10 cases)"

2. STRONG NEGATIVE PATTERNS
   - Topics that appear FREQUENTLY in IRRELEVANT papers but RARELY/NEVER in RELEVANT papers
   - Example: "User consistently rejects papers about AGN and quasars (e.g. 1/12 relevant)"

3. NUANCED PATTERNS
   - More subtle preferences that depend on context
   - Example: "User likes stellar population papers ONLY in galactic context (5 relevant), not in star clusters (0 relevant)"

For each pattern you identify, you MUST provide:
- Clear description of the pattern
- Confidence score (0.0-1.0): How consistent is this pattern? Only suggest patterns with confidence > {CONFIDENCE_THRESHOLD}
- Evidence: Count of papers showing this pattern
- Suggested text: Exact wording to add to the user's prompt

CRITICAL: Only include patterns you are very confident about (>{int(CONFIDENCE_THRESHOLD*100)}% consistency).

Respond as valid JSON with this exact structure:
{{
    "strong_positive": [
        {{
            "pattern": "description of what user likes",
            "confidence": 0.85,
            "evidence": "7/8 JWST papers marked relevant",
            "suggested_addition": "Papers using JWST near-infrared observations"
        }}
    ],
    "strong_negative": [
        {{
            "pattern": "description of what user dislikes",
            "confidence": 0.92,
            "evidence": "1/12 AGN papers marked relevant",
            "suggested_addition": "Not interested in: Active galactic nuclei (AGN) or quasars"
        }}
    ],
    "nuanced": [
        {{
            "pattern": "description of contextual preference",
            "confidence": 0.78,
            "evidence": "5/5 stellar pop papers in galaxies relevant, 0/3 in clusters",
            "suggested_nuance": "Stellar populations in galactic context, not star clusters"
        }}
    ]
}}

Only include patterns with high confidence. If no strong patterns exist, return empty arrays."""

def detect_voting_patterns_with_ai(user_id, openai_client):
    """
    Use GPT-4 to analyze voting patterns and suggest prompt improvements.
    Uses random sampling to avoid recency bias.
    """
    print(f"\n Analyzing voting patterns for user {user_id}...")
    
    # 1. Fetch all voting history
    try:
        votes_response = supabase.table('paper_feedback')\
            .select('paper_title, paper_arxiv_id, paper_abstract, vote, created_at')\
            .eq('user_id', user_id)\
            .execute()
        
        votes = votes_response.data
        
        if len(votes) < MIN_VOTES_FOR_ANALYSIS:
            print(f"   Not enough votes: {len(votes)}/{MIN_VOTES_FOR_ANALYSIS} minimum")
            return None
        
        print(f"   Found {len(votes)} total votes")
        
    except Exception as e:
        print(f"   Database error: {e}")
        return None
    
    # 2. Separate relevant vs irrelevant
    relevant_papers = [v for v in votes if v['vote'] == 'up']
    irrelevant_papers = [v for v in votes if v['vote'] == 'down']
    
    print(f"   {len(relevant_papers)} relevant, {len(irrelevant_papers)} not relevant")
    
    # 3. Check diversity requirements
    if len(relevant_papers) < MIN_RELEVANT_VOTES or len(irrelevant_papers) < MIN_NOT_RELEVANT_VOTES:
        print(f"   Need more diverse votes")
        return None
    
    # 4. Random sample from voted papers for analysis
    sampled_relevant = random.sample(
        relevant_papers,
        min(MAX_PAPERS_FOR_ANALYSIS, len(relevant_papers))
    )
    sampled_irrelevant = random.sample(
        irrelevant_papers,
        min(MAX_PAPERS_FOR_ANALYSIS, len(irrelevant_papers))
    )
    
    print(f"   Analyzing random sample: {len(sampled_relevant)} relevant, {len(sampled_irrelevant)} not relevant")
    
    # 5. Format papers for GPT analysis
    relevant_formatted = _format_papers_for_analysis(
        sampled_relevant,
        limit=MAX_PAPERS_FOR_ANALYSIS
    )
    irrelevant_formatted = _format_papers_for_analysis(
        sampled_irrelevant,
        limit=MAX_PAPERS_FOR_ANALYSIS
    )
    
    # 5. Build analysis prompt
    analysis_prompt = _build_pattern_analysis_prompt(
        relevant_papers, 
        irrelevant_papers,
        relevant_formatted, 
        irrelevant_formatted
    )
    
    # 6. Ask GPT-4 to analyze patterns
    try:
        print(f"   Sending to GPT-4 for analysis...")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        patterns = json.loads(response.choices[0].message.content)
        
        # Count high-confidence patterns
        total_patterns = (
            len([p for p in patterns.get('strong_positive', []) if p['confidence'] > CONFIDENCE_THRESHOLD]) +
            len([p for p in patterns.get('strong_negative', []) if p['confidence'] > CONFIDENCE_THRESHOLD]) +
            len([p for p in patterns.get('nuanced', []) if p['confidence'] > CONFIDENCE_THRESHOLD])
        )
        
        print(f"   Found {total_patterns} high-confidence patterns")
        return patterns
        
    except openai.APIError as e:
        print(f"   OpenAI API error: {e}")
        return None
        
    except json.JSONDecodeError as e:
        print(f"   Error parsing GPT response: {e}")
        return None
        
    except Exception as e:
        print(f"   Unexpected error: {e}")
        return None

def _store_pattern_suggestion(user_id, pattern_type, pattern, suggestion_key, current_prompt):
    """
    Store a single pattern suggestion in the database.
    
    Args:
        user_id: User's id
        pattern_type: 'positive', 'negative', or 'nuanced'
        pattern: Pattern dictionary from GPT
        suggestion_key: Key to extract suggestion text ('suggested_addition' or 'suggested_nuance')
        current_prompt: User's current prompt
    
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        supabase.table('prompt_suggestions').insert({
            'user_id': user_id,
            'pattern_type': pattern_type,
            'pattern_description': pattern.get('pattern', ''),
            'confidence': pattern.get('confidence', 0),
            'evidence': pattern.get('evidence', ''),
            'suggested_text': pattern.get(suggestion_key, ''),
            'current_prompt': current_prompt,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }).execute()
        
        return True
        
    except Exception as e:
        pattern_desc = pattern.get('pattern', 'Unknown')[:30]
        print(f"      Warning: Could not store pattern '{pattern_desc}': {e}")
        return False

def create_prompt_suggestions_from_patterns(user_id, patterns, current_prompt):
    """
    Convert AI-detected patterns into database suggestions.
    
    Args:
        user_id: User's id
        patterns: Dictionary with 'strong_positive', 'strong_negative', 'nuanced' arrays
        current_prompt: User's current prompt text
    
    Returns:
        Number of suggestions successfully created
    """
    suggestions_created = 0
    
    # Process each pattern type
    pattern_types = [
        ('strong_positive', 'positive', 'suggested_addition'),
        ('strong_negative', 'negative', 'suggested_addition'),
        ('nuanced', 'nuanced', 'suggested_nuance'),
    ]
    
    for patterns_key, db_type, suggestion_key in pattern_types:
        for pattern in patterns.get(patterns_key, []):
            # Skip low-confidence patterns
            if pattern.get('confidence', 0) <= CONFIDENCE_THRESHOLD:
                continue
            
            # Create suggestion entry
            success = _store_pattern_suggestion(
                user_id=user_id,
                pattern_type=db_type,
                pattern=pattern,
                suggestion_key=suggestion_key,
                current_prompt=current_prompt
            )
            
            if success:
                suggestions_created += 1
                pattern_desc = pattern.get('pattern', 'Unknown')[:50]
                print(f"      {db_type.capitalize()} pattern: {pattern_desc}...")
    
    return suggestions_created

def _get_user_vote_statistics(user_id):
    """
    Get vote statistics for a user.
    
    Args:
        user_id: User's id
    
    Returns:
        Dictionary with 'total', 'relevant', 'not_relevant' counts, or None on error
    """
    try:
        votes_response = supabase.table('paper_feedback')\
            .select('vote')\
            .eq('user_id', user_id)\
            .execute()
        
        votes = votes_response.data
        
        return {
            'total': len(votes),
            'relevant': sum(1 for v in votes if v['vote'] == 'up'),
            'not_relevant': sum(1 for v in votes if v['vote'] == 'down')
        }
        
    except Exception as e:
        print(f"   Error fetching votes: {e}")
        return None


def _is_analysis_cooldown_passed(user):
    """
    Check if enough time has passed since last analysis attempt.
    
    Args:
        user: User profile dictionary
    
    Returns:
        True if cooldown period has passed or no previous attempt
    """
    last_attempt = user.get('last_analysis_attempt')
    
    if not last_attempt:
        return True
    
    try:
        last_attempt_date = datetime.fromisoformat(last_attempt.replace('Z', ''))
        days_since = (datetime.now() - last_attempt_date).days
        return days_since >= ANALYSIS_COOLDOWN_DAYS
        
    except Exception as e:
        print(f"   Warning: Could not parse last_analysis_attempt date: {e}")
        return True

def _days_until_next_analysis(user):
    """
    Calculate days remaining in cooldown period.
    
    Args:
        user: User profile dictionary
    
    Returns:
        Number of days until next analysis allowed
    """
    last_attempt = user.get('last_analysis_attempt')
    
    if not last_attempt:
        return 0
    
    try:
        last_attempt_date = datetime.fromisoformat(last_attempt.replace('Z', ''))
        days_since = (datetime.now() - last_attempt_date).days
        return max(0, ANALYSIS_COOLDOWN_DAYS - days_since)
        
    except:
        return 0

def _has_pending_suggestions(user_id):
    """
    Check if user has any pending suggestions.
    
    Args:
        user_id: User's id
    
    Returns:
        True if pending suggestions exist
    """
    try:
        response = supabase.table('prompt_suggestions')\
            .select('id', count='exact')\
            .eq('user_id', user_id)\
            .eq('status', 'pending')\
            .execute()
        
        return response.count > 0
        
    except:
        return False

def _count_pending_suggestions(user_id):
    """Get count of pending suggestions for user"""
    try:
        response = supabase.table('prompt_suggestions')\
            .select('id', count='exact')\
            .eq('user_id', user_id)\
            .eq('status', 'pending')\
            .execute()
        
        return response.count
        
    except:
        return 0

def _update_analysis_timestamp(user_id, days_offset=0):
    """
    Record analysis attempt timestamp for a user.
    
    Args:
        user_id: User's id
        days_offset: Days to add/subtract from current time (for cooldowns)
    
    Returns:
        True if successful
    """
    try:
        timestamp = datetime.now() + timedelta(days=days_offset)
        
        supabase.table('user_profiles').update({
            'last_analysis_attempt': timestamp.isoformat()
        }).eq('id', user_id).execute()
        
        return True
        
    except Exception as e:
        print(f"   Warning: Could not update analysis timestamp: {e}")
        return False

def check_and_evolve_prompts():
    """
    Agent that checks all active users and suggests prompt improvements.
    
    Runs daily as part of the newsletter workflow. For each user:
    1. Checks if they have sufficient votes (min 20 total, diverse)
    2. Checks if analysis cooldown period has passed (14 days)
    3. Checks for existing pending suggestions
    4. Analyzes voting patterns with GPT-4
    5. Creates suggestions in database
    
    Cooldown logic:
    - Ssuggestions created: Wait 14 days
    - No patterns found: Wait 14 days
    - Failure: Wait 1 day
    
    Returns:
        Dictionary with summary statistics
    """
    print("\n" + "="*60)
    print("Agent: Checking for learning opportunities...")
    print("="*60)
    
    # Load all active users
    try:
        users_response = supabase.table('user_profiles')\
            .select('*')\
            .eq('active', True)\
            .execute()
        users = users_response.data
    except Exception as e:
        print(f"Error loading users: {e}")
        return {'error': 'Could not load users', 'suggestions_made': 0}
    
    print(f"Checking {len(users)} active users...\n")
    
    # Track statistics
    stats = {
        'users_checked': 0,
        'users_skipped_votes': 0,
        'users_skipped_cooldown': 0,
        'users_skipped_pending': 0,
        'users_analyzed': 0,
        'suggestions_made': 0,
        'analysis_failures': 0,
        'no_patterns_found': 0
    }
    
    for user in users:
        stats['users_checked'] += 1
        print(f"[{stats['users_checked']}/{len(users)}] Checking {user['email']}...")
        
        # Check 1: Is there enough votes
        vote_stats = _get_user_vote_statistics(user['id'])
        
        if not vote_stats:
            print(f"   Could not fetch vote data")
            stats['users_skipped_votes'] += 1
            continue
        
        if vote_stats['total'] < MIN_VOTES_FOR_ANALYSIS:
            print(f"   Not enough votes: {vote_stats['total']}/{MIN_VOTES_FOR_ANALYSIS}")
            stats['users_skipped_votes'] += 1
            continue
        
        if vote_stats['relevant'] < MIN_RELEVANT_VOTES or vote_stats['not_relevant'] < MIN_NOT_RELEVANT_VOTES:
            print(f"   Not diverse: {vote_stats['relevant']} relevant, {vote_stats['not_relevant']} not relevant")
            print(f"      (need {MIN_RELEVANT_VOTES}+ relevant, {MIN_NOT_RELEVANT_VOTES}+ not relevant)")
            stats['users_skipped_votes'] += 1
            continue
        
        # Check 2: Cooldown period
        if not _is_analysis_cooldown_passed(user):
            days_left = _days_until_next_analysis(user)
            print(f"   Cooldown active: {days_left} days remaining")
            stats['users_skipped_cooldown'] += 1
            continue
        
        # Check 3: Existing pending suggestions
        if _has_pending_suggestions(user['id']):
            pending_count = _count_pending_suggestions(user['id'])
            print(f"   Already has {pending_count} pending suggestions")
            stats['users_skipped_pending'] += 1
            continue
        
        # All checks passed - run analysis
        print(f"   Ready for analysis ({vote_stats['total']} votes)")
        
        # Run pattern detection
        patterns = detect_voting_patterns_with_ai(user['id'], openai_client)
        stats['users_analyzed'] += 1
        
        if patterns:
            # GPT analysis succeeded - create suggestions
            count = create_prompt_suggestions_from_patterns(
                user['id'],
                patterns,
                user['custom_prompt']
            )
            
            if count > 0:
                # SUCCESS - created suggestions
                # Set cooldown to prevent re-analysis for 14 days
                _update_analysis_timestamp(user['id'])
                print(f"   Created {count} suggestions")
                print(f"      Next analysis in {ANALYSIS_COOLDOWN_DAYS} days")
                stats['suggestions_made'] += count
            else:
                # No high-confidence patterns found
                # Set cooldown - need more votes to accumulate
                _update_analysis_timestamp(user['id'])
                print(f"   No high-confidence patterns found")
                print(f"      Next analysis in {ANALYSIS_COOLDOWN_DAYS} days (waiting for more votes)")
                stats['no_patterns_found'] += 1
        else:
            # Analysis failed
            # Set short cooldown - retry tomorrow
            offset = ANALYSIS_COOLDOWN_ON_FAILURE - ANALYSIS_COOLDOWN_DAYS  # -13 days
            _update_analysis_timestamp(user['id'], days_offset=offset)
            print(f"   Pattern detection failed")
            print(f"      Will retry in {ANALYSIS_COOLDOWN_ON_FAILURE} day")
            stats['analysis_failures'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Agent Summary:")
    print(f"  Users checked: {stats['users_checked']}")
    print(f"  Users analyzed: {stats['users_analyzed']}")
    print(f"  Suggestions created: {stats['suggestions_made']}")
    print(f"  No patterns found: {stats['no_patterns_found']}")
    print(f"  Analysis failures: {stats['analysis_failures']}")
    print(f"  Skipped (insufficient votes): {stats['users_skipped_votes']}")
    print(f"  Skipped (cooldown): {stats['users_skipped_cooldown']}")
    print(f"  Skipped (pending suggestions): {stats['users_skipped_pending']}")
    print("="*60)
    
    return stats

# ============================================
# EMAIL TEMPLATE
# ============================================

def _generate_user_token(subscriber):
    """
    Generate secure token for user management links.
    
    Args:
        subscriber: User profile dictionary
    
    Returns:
        32-character token string
    """
    token_input = f"{subscriber['id']}{subscriber['email']}{subscriber.get('created_at', '')}"
    return hashlib.sha256(token_input.encode()).hexdigest()[:32]

def _format_review_section_headers(review_text):
    """
    Format section headers in review text with italic styling.
        
    Args:
        review_text: Plain text review from GPT
    
    Returns:
        HTML-formatted review with styled headers
    """
    section_headers = ['Paper Overview', 'Methodology', 'Main Findings', 'Relevance to Your Prompt', 'Limitations']
    
    for header in section_headers:
        # Match header at start of line
        pattern = rf'(^|\n)\s*({re.escape(header)}):?\s*(\n|$)'
        replacement = rf'\1<i style="display: block; margin-top: 24px; margin-bottom: 0; font-weight: 600;">\2</i>'
        review_text = re.sub(pattern, replacement, review_text, flags=re.MULTILINE)
    
    # Convert newlines to HTML breaks
    review_text = review_text.replace('\n', '<br>')
    
    return review_text

def create_email_html(relevant_papers, subscriber, send_empty=True):
    """    
    Generate HTML email for newsletter.
    
    Args:
        relevant_papers: List of paper dictionaries with title, authors, review, links
        subscriber: User profile dictionary with id, email, full_name, created_at
        send_empty: If True, send emails even when no papers found
    
    Returns:
        HTML string for email body, or None if no papers and send_empty=False
    """
    
    date = datetime.now().strftime('%d %B %Y')
    subscriber_name = subscriber['full_name']
    user_id = subscriber['id']
    
    # Create secure token for management links
    token = _generate_user_token(subscriber)
    settings_link = f"https://promptsandpapers.com/?action=dashboard&token={token}"
    unsubscribe_link = f"https://promptsandpapers.com/?action=management&token={token}"
    dashboard_link = f"https://promptsandpapers.com/?view=recent-papers"
    
    # Handle empty newsletters - separate templates (need to update both if any design changes)
    if not relevant_papers:
        if not send_empty:
            return None
            
        html = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
            <div style="background: #1976D2; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
                <h1 style="margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">Your Paper Selection by Prompts & Papers</h1>
                <p style="margin: 0; font-size: 16px; opacity: 0.9;">Hello {subscriber_name}!</p>
            </div>
            <div style="background: white; padding: 30px; border-radius: 8px; text-align: center;">
                <h3 style="color: #666; margin-top: 0;">No relevant papers found today</h3>
                <p style="color: #888;">We scanned today's submissions but didn't find any papers matching your criteria.</p>
            </div>
        </body>
        </html>
        """
        return html
    
    # Build papers HTML
    papers_html = ""
    for i, paper in enumerate(relevant_papers, 1):
        paper_border_color = "#1976D2"
        paper_number = i
        title = paper['title']
        arxiv_id = paper['arxiv_id']
        
        # Create links for voting
        recent_papers_link = f"https://promptsandpapers.com/?view=recent-papers&filter=today&user={subscriber['id']}"
        
        # Format review text
        review_text = _format_review_section_headers(paper.get('review', ''))

        papers_html += f"""
        <div style="background: white; border-left: 4px solid {paper_border_color}; padding: 25px; margin-bottom: 20px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <table width="100%" cellpadding="0" cellspacing="0" style="border-bottom: 1px solid #e5e7eb; padding-bottom: 15px; margin-bottom: 15px;">
                <tr>
                    <td style="vertical-align: top; padding-right: 15px;">
                        <h3 style="margin: 0; color: #0f172a; font-size: 18px; line-height: 1.4;">
                            {paper_number}. {title}
                        </h3>
                        <p style="margin: 0 0 5px 0; font-size: 14px; color: #0f172a;">
                            <strong>Authors:</strong> {', '.join(paper['authors'])}
                        </p>
                        <p style="margin: 0; font-size: 13px; color: #0f172a;">
                            <strong>arXiv ID:</strong> {paper['arxiv_id']}
                        </p>
                    </td>
                    <td style="vertical-align: top; text-align: right; width: 30%;">
                        <!-- Button container table for better Outlook support -->
                        <table cellpadding="0" cellspacing="0" style="margin: 0 0 0 auto;">
                            <tr>
                                <td style="padding: 0 2px;">
                                    <a href="{recent_papers_link}" style="display: block; width: 36px; height: 36px; background: #0f172a; color: white; border: 2px solid #0f172a; text-decoration: none; border-radius: 4px; font-size: 18px; line-height: 32px; text-align: center; font-weight: bold; box-sizing: border-box;">+</a>
                                </td>
                                <td style="padding: 0 2px;">
                                    <a href="{recent_papers_link}" style="display: block; width: 36px; height: 36px; box-sizing: border-box; background: white; color: #0f172a; border: 2px solid #0f172a; text-decoration: none; border-radius: 4px; font-size: 18px; line-height: 32px; text-align: center; font-weight: bold;">-</a>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
            
            <div style="color: #0f172a; font-size: 15px; line-height: 1.7;">
                {review_text}
            </div>

            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 14px; color: #333;">
                    <strong>Open the paper at:</strong>
                    <a href="{paper['arxiv_link']}" style="color: #1976D2; text-decoration: none; margin-left: 8px;">arXiv</a>
                    <span style="margin: 0 8px; color: #d1d5db;">•</span>
                    <a href="{paper.get('pdf_link', paper['arxiv_link'].replace('abs', 'pdf'))}" style="color: #1976D2; text-decoration: none;">PDF</a>
                    <span style="margin: 0 8px; color: #d1d5db;">•</span>
                    <a href="https://www.alphaxiv.org/abs/{arxiv_id}" style="color: #1976D2; text-decoration: none;">alphaXiv</a>
                    <span style="margin: 0 8px; color: #d1d5db;">•</span>
                    <a href="https://www.zotero.org/save?q={paper['arxiv_link']}" style="color: #1976D2; text-decoration: none;">Zotero</a>
                </p>
            </div>
        </div>
        """
    
    # Complete HTML
    html = f"""
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
        <div style="background: #1976D2; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
            <h1 style="margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">Your Paper Selection</h1>
            <p style="margin: 0; font-size: 16px; opacity: 0.9;">{date} • {len(relevant_papers)} papers</p>
        </div>
        
        {papers_html}
        
        <div style="background: white; padding: 25px; margin-top: 30px; border-radius: 8px; text-align: center; font-size: 13px; color: #6b7280;">
            <p style="margin: 0 0 10px 0;">Prompts & Papers - Your personalized research newsletter</p>
            <p style="margin: 0;">
                <a href="{settings_link}" style="color: #1976D2; text-decoration: none;">Your Account</a> | 
                <a href="{unsubscribe_link}" style="color: #1976D2; text-decoration: none;">Unsubscribe</a>
            </p>
        </div>
    </body>
    </html>
    """
    return html

def send_email(to_email, subject, html_content):
    """
    Send HTML email via SendGrid API with automatic retry on server errors.
    
    Args:
        to_email: User email address
        subject: Email subject
        html_content: HTML email body
    
    Returns:
        Tuple of (success: bool, result: int|str)
    """
    # Validation
    if not SENDGRID_KEY or not FROM_EMAIL:
        return False, "Email not configured"
    
    if not html_content or not html_content.strip():
        return False, "Empty email content"
    
    if not to_email or '@' not in to_email:
        return False, "Invalid recipient email"
    
    # Attempt to send with retry
    last_error = None
    
    for attempt in range(MAX_EMAIL_RETRIES):
        try:
            message = Mail(
                from_email=FROM_EMAIL,
                to_emails=to_email,
                subject=subject,
                html_content=html_content
            )
            
            sg = SendGridAPIClient(SENDGRID_KEY)
            response = sg.send(message)
            
            return True, response.status_code
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Retry on server errors (5xx) or network issues
            should_retry = (
                '500' in error_str or 
                '502' in error_str or 
                '503' in error_str or 
                'timeout' in error_str.lower() or
                'connection' in error_str.lower()
            )
            
            if should_retry and attempt < MAX_EMAIL_RETRIES - 1:
                wait_time = 2 ** attempt  # 1s, 2s
                print(f"   Retrying in {wait_time}s... (attempt {attempt + 2}/{MAX_EMAIL_RETRIES})")
                time.sleep(wait_time)
                continue
            
            # Don't retry
            break
    
    # All attempts failed
    error_type = type(last_error).__name__
    print(f"   Failed to send email ({error_type}): {last_error}")
    return False, f"{error_type}: {str(last_error)}"

# ============================================
# NEWSLETTER PROCESSING
# ============================================

def _parse_timestamp_safe(timestamp_str):
    """
    Parse ISO timestamp from Supabase, converting to datetime.date.
    
    Handles timezone indicators (Z, +00:00) and microsecond precision variations
    that Supabase may return but Python's fromisoformat() requires in specific format.
    
    Args:
        timestamp_str: ISO format timestamp from database
    
    Returns:
        datetime.date object, or None if parsing fails
    """
    if not timestamp_str:
        return None
    
    try:
        # Remove timezone indicators
        cleaned = timestamp_str.split('+')[0].split('Z')[0]
        
        # Normalize microseconds to exactly 6 digits
        if '.' in cleaned:
            date_part, micro_part = cleaned.rsplit('.', 1)
            micro_part = micro_part.ljust(6, '0')[:6]
            cleaned = f"{date_part}.{micro_part}"
        
        return datetime.fromisoformat(cleaned).date()
        
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}")
        return None

def should_send_newsletter_today(subscriber):
    """
    Determine if subscriber should receive newsletter today.
    
    Rules:
    - Skip weekends
    - Daily: Send if 1+ days since last email
    - Weekly: Send on preferred day if 7+ days since last email
    - Never sent before: Always send (subject to day/frequency rules)
    
    Args:
        subscriber: Dict with frequency, preferred_day, last_sent, email
    
    Returns:
        True if newsletter should be sent today
    """
    frequency = subscriber.get('frequency', '').lower()
    
    # Validate frequency
    if frequency not in ['daily', 'weekly']:
        print(f"   Warning: Invalid frequency '{subscriber.get('frequency')}'")
        return False
    
    today = datetime.now()
    
    # Skip weekends
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Calculate days since last email
    last_sent_date = _parse_timestamp_safe(subscriber.get('last_sent'))
    
    if last_sent_date:
        days_since_last = (today.date() - last_sent_date).days
    else:
        # Never sent before - use infinity so first email is sent
        days_since_last = float('inf')
    
    # Apply frequency rules
    if frequency == 'daily':
        return days_since_last >= 1
    
    elif frequency == 'weekly':
        preferred_day = subscriber.get('preferred_day', 1)  # Database: 1=Monday, 7=Sunday
        today_day = today.weekday() + 1  # Convert Python's 0-6 to 1-7
        return today_day == preferred_day and days_since_last >= 7
    
    return False

# ============================================
# PAPER CLASSIFICATION
# ============================================

def _create_classification_filter(user_prompt, openai_client):
    """
    Create a classification function using a user's custom prompt.
    
    Args:
        user_prompt: User's custom research interests/criteria
        openai_client: Initialized OpenAI client
    
    Returns:
        Function that classifies papers based on user's specific criteria
    """
    def classify_paper(title, abstract):
        """Classify if paper is relevant to user's specific interests"""
        
        formatted_prompt = f"""You are a researcher evaluating whether an arXiv paper is relevant based on the user's research interests.

{user_prompt}

Paper to evaluate:
Title: {title}
Abstract: {abstract}

Respond as JSON:
{{"is_relevant": true/false,
"confidence": 0.0-1.0,
"reasoning": "Brief explanation"
}}"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.0,
                seed=42,
                timeout=30,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"    Error in classification: {e}")
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    return classify_paper

def process_subscriber(subscriber, scraped_papers):
    """
    Process one subscriber: classify papers and generate reviews.
    
    Args:
        subscriber: User profile dict containing:
            - full_name: Display name
            - email: Email address
            - custom_prompt: Research interests for classification
            - arxiv_category: arXiv category (default: 'astro-ph.GA')
        scraped_papers: Dict of {category: [papers]} from scraping step
    
    Returns:
        List of relevant paper dicts with reviews, sorted by confidence
    """
    print(f"\n[User] {subscriber['full_name']} ({subscriber['email']})")
    
    # Get user's category
    category = subscriber.get('arxiv_category', 'astro-ph.GA')
    
    # Get pre-scraped papers for this category
    papers = scraped_papers.get(category, [])
    
    if not papers:
        print(f"   No papers found for category {category}")
        return []
    
    print(f"  Processing {len(papers)} papers from {category}")
    
    # Create classification function with the user's custom prompt
    classify_paper = _create_classification_filter(
        subscriber['custom_prompt'],
        openai_client
    )
    
    # Initialize scraper (for PDF download and review generation)
    scraper = ArxivWebScraper(OPENAI_KEY)
    
    # Classify and review relevant papers
    relevant_papers = []
    
    for paper in papers:
        try:
            # Validate required fields
            if not all(key in paper for key in ['arxiv_id', 'title', 'abstract']):
                continue
            
            # Classify paper using the user's criteria
            classification = classify_paper(paper['title'], paper['abstract'])
            
            # Log classification result
            status = "✓" if classification['is_relevant'] else "✗"
            print(f"    {status} {paper['arxiv_id']}: conf={classification['confidence']:.2f}")
            
            # Skip if not relevant
            if not classification['is_relevant']:
                continue
            
            # Generate review
            if READ_FULL_PDFS:
                full_text = scraper.download_pdf(
                    paper.get('pdf_url', paper['link'].replace('abs', 'pdf')),
                    paper['arxiv_id']
                )
                
                if full_text:
                    review = scraper.generate_full_paper_review(
                        paper['title'],
                        paper['abstract'],
                        full_text
                    )
                else:
                    # PDF failed - fall back to abstract
                    review = scraper.generate_abstract_summary(
                        paper['title'],
                        paper['abstract']
                    )
            else:
                # Abstract-only review
                review = scraper.generate_abstract_summary(
                    paper['title'],
                    paper['abstract']
                )
            
            # Format author list (truncate if too many)
            author_list = paper['authors'][:MAX_AUTHORS_DISPLAY]
            if len(paper['authors']) > MAX_AUTHORS_DISPLAY:
                author_list = author_list + ['et al.']
            
            # Build paper result
            relevant_papers.append({
                'title': paper['title'],
                'arxiv_id': paper['arxiv_id'],
                'authors': author_list,
                'abstract': paper['abstract'],
                'review': review,
                'arxiv_link': paper['link'],
                'pdf_link': paper.get('pdf_url', paper['link'].replace('abs', 'pdf')),
                'confidence': classification['confidence'],
                'reasoning': classification['reasoning']
            })
            
        except Exception as e:
            print(f"    Error processing {paper.get('arxiv_id', 'unknown')}: {e}")
            continue
    
    # Sort by confidence (highest first)
    relevant_papers.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"  Found {len(relevant_papers)} relevant papers")
    
    return relevant_papers

# ============================================
# PAPER SCRAPING
# ============================================

def scrape_categories_for_subscribers(subscribers):
    """
    Scrape arXiv papers for all unique categories used by subscribers.
    
    This scrapes each category ONCE and caches results, avoiding duplicate
    API calls when multiple users subscribe to the same category.
    
    Args:
        subscribers: List of user profile dicts
    
    Returns:
        Dict of {category: [papers]} with scraped papers for each category
    """
    # Get unique categories from all subscribers
    categories = set()
    for subscriber in subscribers:
        category = subscriber.get('arxiv_category', 'astro-ph.GA')
        categories.add(category)
    
    print(f"\n{'='*60}")
    print(f"Scraping {len(categories)} unique categories")
    print(f"{'='*60}")
    
    scraped_papers = {}
    scraper = ArxivWebScraper(OPENAI_KEY)
    target_date = datetime.now().date()
    
    for i, category in enumerate(sorted(categories), 1):
        print(f"\n[{i}/{len(categories)}] Scraping {category}...")
        
        try:
            papers = scraper.scrape_recent_submissions_by_headers(
                category=category,
                target_date=target_date
            )
            
            scraped_papers[category] = papers
            print(f"  Found {len(papers)} papers in {category}")
            
        except Exception as e:
            print(f"  Error scraping {category}: {e}")
            scraped_papers[category] = []
    
    total_papers = sum(len(papers) for papers in scraped_papers.values())
    print(f"\n{'='*60}")
    print(f"Scraping complete: {total_papers} papers across {len(categories)} categories")
    print(f"{'='*60}")
    
    return scraped_papers

def run_daily_digest(force_send=False):
    """
    Main newsletter workflow - runs daily via GitHub Actions.
    
    Process:
    1. Scrapes arXiv once per category
    2. For each user: classify papers, store in database
    3. Send email if scheduled (based on frequency)
    
    Args:
        force_send: If True, send to all users regardless of schedule
    """    
    print("Loading subscribers...")
    subscribers = load_subscribers()
    print(f"Loaded {len(subscribers)} subscribers")
    
    # Scrape all categories ONCE
    scraped_papers = scrape_categories_for_subscribers(subscribers)
    
    users_processed = 0
    emails_sent = 0
    
    for i, subscriber in enumerate(subscribers, 1):
        print(f"\n[{i}/{len(subscribers)}] {subscriber['full_name']}")
        
        try:
            # Classify papers with the user's custom prompt
            relevant_papers = process_subscriber(subscriber, scraped_papers)
            
            # Store papers in database (happens daily)
            for paper in relevant_papers:
                store_paper(subscriber['id'], paper)
            
            users_processed += 1
            print(f"  Stored {len(relevant_papers)} papers")
            
            # Check if email should be sent today
            if not should_send_newsletter_today(subscriber) and not force_send:
                print(f"  Email not scheduled today")
                continue
            
            # Get papers to send based on frequency
            if subscriber['frequency'] == 'weekly':
                # Weekly: Get last 7 days of papers
                cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
                papers_to_send = supabase.table('sent_papers').select('*')\
                    .eq('user_id', subscriber['id'])\
                    .gte('processed_at', cutoff_date)\
                    .order('processed_at', desc=False).execute().data
            else:
                # Daily: Send today's papers
                papers_to_send = relevant_papers
            
            print(f"  Sending {len(papers_to_send)} papers")
            
            if not papers_to_send and not SEND_EMPTY_DIGESTS:
                continue
            
            # Generate and send email
            html_content = create_email_html(papers_to_send, subscriber, SEND_EMPTY_DIGESTS)
            subject = f"Prompts & Papers - {len(papers_to_send)} papers" if papers_to_send else "Prompts & Papers"
            
            success, result = send_email(subscriber['email'], subject, html_content)
            
            if success:
                print(f"  Email sent!")
                emails_sent += 1
                
                # Update last_sent timestamp
                update_last_sent(subscriber['email'], datetime.now().isoformat())
                
                # Update papers_sent_total counter
                current_total = supabase.table('user_profiles')\
                    .select('papers_sent_total')\
                    .eq('id', subscriber['id'])\
                    .single().execute().data['papers_sent_total']
                
                supabase.table('user_profiles').update({
                    'papers_sent_total': current_total + len(papers_to_send)
                }).eq('id', subscriber['id']).execute()
            else:
                print(f"  Failed: {result}")
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Complete: {users_processed} users processed, {emails_sent} emails sent")
    print(f"{'='*60}")

# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """
    Main entry point for the newsletter
    
    Runs two workflows:
    1. Newsletter generation and sending (daily)
    2. AI learning agent for prompt evolution
    
    Command-line arguments:
        --force: Send emails to all users regardless of schedule
    
    Returns:
        0 if successful, 1 if errors occurred
    
    Environment variables required:
        OPENAI_KEY, SENDGRID_KEY, FROM_EMAIL,
        SUPABASE_URL, SUPABASE_SERVICE_KEY
    """
    start_time = time.time()
    force_send = '--force' in sys.argv
    
    # Print header
    print("\n" + "=" * 60)
    print("ArXiv Newsletter Agent")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Validate environment variables
    required_vars = {
        'OPENAI_KEY': OPENAI_KEY,
        'SENDGRID_KEY': SENDGRID_KEY,
        'FROM_EMAIL': FROM_EMAIL,
        'SUPABASE_URL': SUPABASE_URL,
        'SUPABASE_SERVICE_KEY': SUPABASE_KEY
    }
    
    missing = [name for name, value in required_vars.items() if not value]
    
    if missing:
        print("\n ERROR: Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\n Set these in GitHub Secrets or .env file")
        return 1
    
    errors = []
    
    # Step 1: Newsletter workflow
    try:
        print("\n Newsletter Workflow")
        print("-" * 60)
        run_daily_digest(force_send=force_send)
        print("✓ Newsletter workflow complete")
    except Exception as e:
        print(f"\n Newsletter workflow failed: {e}")
        errors.append("newsletter")
    
    # Step 2: Learning agent
    try:
        print("\n Learning Agent")
        print("-" * 60)
        check_and_evolve_prompts()
        print(" Learning agent complete")
    except Exception as e:
        print(f"\n Learning agent failed: {e}")
        errors.append("agent")
    
    # Print summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 60)
    if errors:
        print(f" Completed with errors in: {', '.join(errors)}")
    else:
        print(" All tasks completed successfully!")
    print(f" Runtime: {minutes}m {seconds}s")
    print("=" * 60 + "\n")
    
    return 1 if errors else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
