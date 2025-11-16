#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:55:05 2025

@author: zc
"""

import requests
import openai
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import os
import re
import PyPDF2
import io

class ArxivWebScraper:
    def __init__(self, openai_key):
        self.client = openai.OpenAI(api_key=openai_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArxivResearchBot/1.0; Academic Research)'
        })
            

    def scrape_recent_submissions_by_headers(self, category="astro-ph.GA", target_date=None, max_pages=5):
        """Main scraping orchestrator"""
        target_date = self._parse_target_date(target_date)
        
        all_papers = []
        page = 0
        found_target = False
        
        while page < max_pages:
            url = self._build_url(category, page)
            soup = self._fetch_page(url)
            
            if not soup:
                break
            
            papers, found_target, should_continue = self._extract_papers_from_page(
                soup, target_date, found_target
            )
            
            all_papers.extend(papers)
            
            if not should_continue:
                break
            
            page += 1
            time.sleep(1)
        
        print(f"Scraped {len(all_papers)} papers from {target_date}")
        return all_papers
    
    def _parse_target_date(self, target_date):
        """Convert target_date to datetime.date object"""
        
        if target_date is None:
            return datetime.now().date()
            
        elif isinstance(target_date, str):
            return datetime.strptime(target_date, "%Y-%m-%d").date()
            
        return target_date
    
    def _build_url(self, category, page):
        """Build arXiv URL for category and page number"""
        if page == 0:
            return f"https://arxiv.org/list/{category}/recent"
            
        skip = page * 50
        return f"https://arxiv.org/list/{category}/recent?skip={skip}&show=50"
    
    def _fetch_page(self, url):
        """Fetch and parse HTML page"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def _extract_papers_from_page(self, soup, target_date, already_found_target):
        """
        Extract papers from a single page.
        
        Returns:
            (papers_list, found_target_on_this_page, should_continue_pagination)
        """
        papers = []
        current_section_date = None
        found_target_this_page = False
        
        content = soup.find('div', id='dlpage')
        if not content:
            return papers, already_found_target, False
        
        elements = content.find_all(['h3', 'dt'])
        
        for element in elements:
            if element.name == 'h3':
                # Parse date header
                current_section_date = self._parse_date_header(element)
                if current_section_date == target_date:
                    found_target_this_page = True
                    print(f"  ✓ Found target date section: {target_date}")
            
            elif element.name == 'dt' and current_section_date == target_date:
                # Extract paper from target date section
                paper = self._extract_paper(element)
                if paper:
                    papers.append(paper)
        
        # Decide if we should continue to next page
        should_continue = self._should_continue_pagination(
            found_target_this_page, 
            already_found_target, 
            len(papers),
            soup
        )
        
        return papers, found_target_this_page, should_continue
    
    def _parse_date_header(self, h3_element):
        """Parse date from header like 'Wed, 17 Sep 2025'"""
        header_text = h3_element.get_text().strip()
        date_match = re.search(r'(\w+),\s+(\d{1,2})\s+(\w+)\s+(\d{4})', header_text)
        
        if date_match:
            day_name, day, month_name, year = date_match.groups()
            try:
                date_str = f"{day} {month_name} {year}"
                return datetime.strptime(date_str, "%d %b %Y").date()
            except ValueError:
                pass
        return None
    
    def _extract_paper(self, dt_element):
        """Extract paper details from dt/dd pair"""
        try:
            # Extract arXiv ID
            arxiv_link = dt_element.find('a', title='Abstract')
            if not arxiv_link:
                return None
            
            arxiv_id = arxiv_link.text.strip().replace('arXiv:', '')
            
            # Find corresponding dd element
            dd = dt_element.find_next_sibling('dd')
            if not dd:
                return None
            
            # Extract title
            title_div = dd.find('div', class_='list-title')
            title = title_div.get_text().replace('Title:', '').strip() if title_div else ""
            
            # Extract authors
            authors_div = dd.find('div', class_='list-authors')
            authors = []
            if authors_div:
                authors = [a.get_text().strip() for a in authors_div.find_all('a')]
            
            # Extract abstract
            abstract_p = dd.find('p', class_='mathjax')
            abstract = abstract_p.get_text().strip() if abstract_p else ""
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'link': f"https://arxiv.org/abs/{arxiv_id}",
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            }
        except Exception as e:
            print(f"  Warning: Could not extract paper: {e}")
            return None
    
    def _should_continue_pagination(self, found_this_page, found_before, papers_count, soup):
        """Decide if we should check the next page"""
        # If we found papers on this page, might be more on next page
        if papers_count > 0:
            return True
        
        # If we found target before but not this page, we're past it - stop
        if found_before and not found_this_page:
            return False
        
        # If we haven't found target yet, keep looking
        if not found_before and not found_this_page:
            # But check if there are more pages (less than 50 papers = last page)
            dt_tags = soup.find_all('dt')
            return len(dt_tags) >= 50
        
        return False

    def download_pdf(self, pdf_url, arxiv_id, max_chars=50000):
        """
        Download and extract text from arXiv PDF.
        
        Args:
            pdf_url: Direct link to PDF
            arxiv_id: Paper ID for logging
            max_chars: Maximum characters to extract (cost control, ~12-15k tokens)
        
        Returns:
            Extracted text string, or None if download fails
        """
        try:
            print(f"  Downloading PDF for {arxiv_id}...")
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            # Parse PDF
            pdf_file = io.BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Stop if we have enough text (for cost control)
                    if len(text) > max_chars:
                        print(f"    Truncated at {page_num + 1} pages ({len(text)} chars)")
                        break
                        
                except Exception as e:
                    print(f"    Warning: Could not extract page {page_num + 1}")
                    continue
            
            if text.strip():
                print(f"    Extracted {len(text)} characters from PDF")
                return text
            else:
                print(f"    Warning: No text extracted from PDF")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"  Network error downloading {arxiv_id}: {e}")
            return None
            
        except PyPDF2.errors.PdfReadError as e:
            print(f"  PDF parse error for {arxiv_id}: {e}")
            return None
            
        except Exception as e:
            print(f"  Unexpected error downloading {arxiv_id}: {e}")
            return None

   
    def generate_full_paper_review(self, title, abstract, full_text):
        """Generate comprehensive review from full paper text"""
                
        # Truncate full text if too long (for cost control)
        if len(full_text) > 40000:
            full_text = full_text[:40000] + "\n\n[Text truncated for length]"
        
        prompt = f"""
        Create a comprehensive review for a scientist based on their research interests. Do NOT repeat the paper title or list authors in your summary - jump straight into the content.
        
        Title: {title}
        Abstract: {abstract}
        
        Full Paper Text:
        {full_text}
        
        Provide a detailed review with these sections:
        
        Paper Overview
        Brief summary of what this paper accomplishes and why it matters.
        
        Methodology
        Important methods, data sources, and analytical approaches used.
        
        Main Findings
        The paper's primary results, measurements, and conclusions.
        
        Relevance to Your Prompt
        Explain how this work connects to the users' research interests described in the prompt.
        
        
        Be thorough but focused. This is for a science newsletter for researchers, so make it informative and engaging. Use plain text for section names - no bold, no bullets, no emojis or special symbols, no special formatting. Write clear paragraphs for each section.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=42
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating full review: {e}")
            # Fallback to abstract-only summary
            return self.abstract_summary(title, abstract)

    def generate_abstract_summary(self, title, abstract):
        """Generate summary from abstract only (fallback)"""
                
        prompt = f"""
        Create a comprehensive review for a scientist based on their research interests. Do NOT repeat the paper title or list authors in your summary - jump straight into the content.
        
        Title: {title}
        Abstract: {abstract}
        
        Provide a detailed review with these sections:
        
        Paper Overview
        Brief summary of what this paper accomplishes and why it matters.
        
        Methodology
        Important methods, data sources, and analytical approaches used.
        
        Main Findings
        The paper's primary results, measurements, and conclusions.
        
        Relevance to Your Prompt
        Explain how this work connects to the users' research interests described in the prompt.
        
        Be thorough but focused. This is for a science newsletter for researchers, so make it informative and engaging. Use plain text for section names - no bold, no bullets, no emojis or special symbols, no special formatting. Write clear paragraphs for each section.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=42
        )
        
        return response.choices[0].message.content.strip()
