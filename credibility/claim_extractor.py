"""
Claim Extractor Module for Claim-Based Verification

Extracts entities, keywords, and context from user claims/questions
to enable evidence-based credibility assessment.

This module uses LOCAL NLP only (no AI APIs):
- Regex patterns for entity extraction
- Keyword extraction for evidence matching
- Indian public figure detection

NOT LIMITED TO SPECIFIC EVENT TYPES - extracts all meaningful keywords.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import string
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Common stop words to filter
STOP_WORDS = {
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then',
    'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'can', 'will', 'just',
    'should', 'now', 'it', 'its', 'this', 'that', 'these',
    'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'any', 'both', 'could',
    'would', 'may', 'might', 'must', 'shall', 'whether',
    'true', 'false', 'really', 'actually', 'happened',
    'news', 'report', 'said', 'says', 'told', 'according',
    'please', 'tell', 'know', 'think', 'believe', 'heard',
    'rumor', 'rumour', 'claim', 'claims', 'alleged', 'allegedly',
}

# Question words to strip from beginning
QUESTION_PREFIXES = [
    r'^is\s+it\s+true\s+that\s+',
    r'^is\s+',
    r'^are\s+',
    r'^was\s+',
    r'^were\s+',
    r'^did\s+',
    r'^does\s+',
    r'^do\s+',
    r'^has\s+',
    r'^have\s+',
    r'^can\s+you\s+verify\s+',
    r'^verify\s+',
    r'^check\s+if\s+',
    r'^is\s+there\s+any\s+truth\s+to\s+',
    r'^what\s+happened\s+to\s+',
    r'^what\s+about\s+',
    r'^tell\s+me\s+about\s+',
]

# Known Indian public figures (for prioritization)
INDIAN_PUBLIC_FIGURES = {
    # Politicians
    'narendra modi', 'rahul gandhi', 'amit shah', 'yogi adityanath',
    'mamata banerjee', 'arvind kejriwal', 'uddhav thackeray',
    'sharad pawar', 'ajit pawar', 'devendra fadnavis', 'eknath shinde',
    'nitish kumar', 'lalu prasad', 'tejashwi yadav', 'akhilesh yadav',
    'mayawati', 'priyanka gandhi', 'sonia gandhi', 'nitin gadkari',
    'rajnath singh', 'nirmala sitharaman', 's jaishankar',
    # Business
    'mukesh ambani', 'gautam adani', 'ratan tata', 'anand mahindra',
    'sundar pichai', 'satya nadella', 'parag agrawal',
    # Entertainment
    'shah rukh khan', 'salman khan', 'aamir khan', 'amitabh bachchan',
    'akshay kumar', 'deepika padukone', 'priyanka chopra', 'ranveer singh',
    'virat kohli', 'ms dhoni', 'sachin tendulkar', 'rohit sharma',
    # Other
    'dalai lama', 'sadhguru', 'baba ramdev',
}

# Indian locations
INDIAN_LOCATIONS = {
    'india', 'delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata',
    'hyderabad', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur',
    'nagpur', 'indore', 'thane', 'bhopal', 'patna', 'vadodara',
    'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
    'rajkot', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad',
    'amritsar', 'allahabad', 'ranchi', 'howrah', 'coimbatore',
    'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'madurai',
    'raipur', 'kota', 'chandigarh', 'guwahati', 'solapur',
    'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'andhra pradesh',
    'uttar pradesh', 'bihar', 'west bengal', 'madhya pradesh', 'rajasthan',
    'gujarat', 'odisha', 'kerala', 'jharkhand', 'assam', 'punjab',
    'chhattisgarh', 'haryana', 'uttarakhand', 'himachal pradesh',
    'tripura', 'meghalaya', 'manipur', 'nagaland', 'goa', 'arunachal pradesh',
    'mizoram', 'sikkim', 'jammu', 'kashmir', 'ladakh',
}


class ClaimExtractor:
    """
    Extracts entities and keywords from user claims for evidence matching.
    
    Uses local regex-based NLP (no AI APIs).
    Flexible - not limited to specific event types.
    """
    
    def __init__(self):
        """Initialize the claim extractor."""
        # Compile question prefix patterns
        self.question_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in QUESTION_PREFIXES
        ]
        
        # Pattern for potential names (capitalized words)
        self.name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        
        # Pattern for quoted text
        self.quote_pattern = re.compile(r'["\'](.*?)["\']')
    
    def extract_claim_info(self, claim: str) -> Dict:
        """
        Extract all relevant information from a claim.
        
        Args:
            claim: User's claim or question
            
        Returns:
            Dictionary with entities, keywords, and metadata
        """
        # Normalize claim
        claim = claim.strip()
        original_claim = claim
        
        # Strip question prefixes
        cleaned_claim = self._strip_question_prefix(claim)
        
        # Extract entities
        entities = self._extract_entities(cleaned_claim)
        
        # Check for Indian context
        is_indian_context = self._is_indian_context(entities, cleaned_claim)
        
        # Extract keywords (all meaningful words)
        keywords = self._extract_keywords(cleaned_claim)
        
        # Generate search queries
        search_queries = self._generate_search_queries(entities, keywords)
        
        return {
            'original_claim': original_claim,
            'cleaned_claim': cleaned_claim,
            'entities': entities,
            'keywords': keywords,
            'search_queries': search_queries,
            'is_indian_context': is_indian_context,
            'indian_figures': [e for e in entities['people'] if self._is_indian_figure(e)],
        }
    
    def _strip_question_prefix(self, claim: str) -> str:
        """Remove question prefixes from claim."""
        result = claim
        for pattern in self.question_patterns:
            result = pattern.sub('', result)
        return result.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Returns dict with:
        - people: Person names
        - places: Location names
        - orgs: Organization names
        - other: Other potential entities
        """
        entities = {
            'people': [],
            'places': [],
            'orgs': [],
            'other': []
        }
        
        text_lower = text.lower()
        
        # Check for known Indian public figures
        for figure in INDIAN_PUBLIC_FIGURES:
            if figure in text_lower:
                # Get proper capitalization from original
                pattern = re.compile(re.escape(figure), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    entities['people'].append(match.group())
                else:
                    entities['people'].append(figure.title())
        
        # Check for Indian locations
        for location in INDIAN_LOCATIONS:
            if re.search(r'\b' + re.escape(location) + r'\b', text_lower):
                entities['places'].append(location.title())
        
        # Extract capitalized name patterns (potential names not in our list)
        for match in self.name_pattern.finditer(text):
            name = match.group().strip()
            name_lower = name.lower()
            # Skip if already found or is a common phrase
            if name_lower not in [p.lower() for p in entities['people']]:
                if name_lower not in STOP_WORDS and len(name) > 3:
                    entities['people'].append(name)
        
        # Extract quoted text as potential entities
        for match in self.quote_pattern.finditer(text):
            quoted = match.group(1).strip()
            if len(quoted) > 2 and quoted.lower() not in STOP_WORDS:
                entities['other'].append(quoted)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        NOT limited to specific event types - extracts all significant words.
        """
        # Remove punctuation except apostrophes
        text_clean = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        # Split into words
        words = text_clean.lower().split()
        
        # Filter stop words and short words
        keywords = []
        for word in words:
            word = word.strip("'")
            if word and word not in STOP_WORDS and len(word) > 2:
                keywords.append(word)
        
        # Count frequency for importance
        word_counts = Counter(keywords)
        
        # Return unique keywords ordered by frequency then appearance
        seen = set()
        result = []
        for word, _ in word_counts.most_common():
            if word not in seen:
                seen.add(word)
                result.append(word)
        
        return result
    
    def _is_indian_context(self, entities: Dict, text: str) -> bool:
        """Check if claim is about Indian context."""
        text_lower = text.lower()
        
        # Check for Indian figures
        for figure in INDIAN_PUBLIC_FIGURES:
            if figure in text_lower:
                return True
        
        # Check for Indian locations
        for location in INDIAN_LOCATIONS:
            if re.search(r'\b' + re.escape(location) + r'\b', text_lower):
                return True
        
        # Check extracted entities
        for person in entities.get('people', []):
            if self._is_indian_figure(person):
                return True
        
        for place in entities.get('places', []):
            if place.lower() in INDIAN_LOCATIONS:
                return True
        
        return False
    
    def _is_indian_figure(self, name: str) -> bool:
        """Check if a name is a known Indian public figure."""
        return name.lower() in INDIAN_PUBLIC_FIGURES
    
    def _generate_search_queries(
        self, 
        entities: Dict[str, List[str]], 
        keywords: List[str]
    ) -> List[str]:
        """
        Generate search queries for evidence retrieval.
        
        Creates multiple query variants for better matching.
        """
        queries = []
        
        # Primary query: all people + top keywords
        people = entities.get('people', [])
        if people:
            # Query with each person
            for person in people[:3]:  # Limit to top 3 people
                # Person + top keywords
                top_kw = [k for k in keywords[:5] if k.lower() not in person.lower()]
                queries.append(f"{person} {' '.join(top_kw[:3])}")
        
        # Add places with keywords
        places = entities.get('places', [])
        if places:
            for place in places[:2]:
                top_kw = [k for k in keywords[:3] if k.lower() != place.lower()]
                if top_kw:
                    queries.append(f"{place} {' '.join(top_kw)}")
        
        # Keywords only query (fallback)
        if keywords:
            queries.append(' '.join(keywords[:5]))
        
        # Deduplicate
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries
    
    def get_entity_variations(self, name: str) -> List[str]:
        """
        Get variations of a name for matching.
        
        E.g., "Ajit Pawar" -> ["Ajit Pawar", "Pawar", "A. Pawar"]
        """
        variations = [name]
        
        parts = name.split()
        if len(parts) >= 2:
            # Last name only
            variations.append(parts[-1])
            # First initial + last name
            variations.append(f"{parts[0][0]}. {parts[-1]}")
            # First name only (if unique enough)
            if len(parts[0]) > 3:
                variations.append(parts[0])
        
        return variations


# Singleton instance
_extractor_instance = None


def get_claim_extractor() -> ClaimExtractor:
    """Get singleton ClaimExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ClaimExtractor()
    return _extractor_instance


# Testing
if __name__ == "__main__":
    print("Testing Claim Extractor")
    print("=" * 60)
    
    extractor = ClaimExtractor()
    
    test_claims = [
        "Is Ajit Pawar dead in a plane crash?",
        "Did Narendra Modi resign as PM?",
        "What happened to Shah Rukh Khan in Mumbai?",
        "Is it true that Adani stocks crashed 50%?",
        "Earthquake in Delhi today?",
        "Did Joe Biden visit India?",
        "Is the new COVID variant spreading in Maharashtra?",
    ]
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        print("-" * 40)
        info = extractor.extract_claim_info(claim)
        print(f"  Entities: {info['entities']}")
        print(f"  Keywords: {info['keywords'][:6]}")
        print(f"  Indian context: {info['is_indian_context']}")
        print(f"  Search queries: {info['search_queries'][:2]}")
    
    print("\n" + "=" * 60)
    print("Claim Extractor module loaded successfully!")
