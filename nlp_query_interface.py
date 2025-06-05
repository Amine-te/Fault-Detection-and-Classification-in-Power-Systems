import re
import streamlit as st
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QueryIntent(Enum):
    SHOW_FAULTS = "show_faults"
    FILTER_FAULTS = "filter_faults"
    EXPLAIN_FAULT = "explain_fault"
    COUNT_FAULTS = "count_faults"
    ANALYZE_SIGNAL = "analyze_signal"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class QueryResult:
    intent: QueryIntent
    entities: Dict[str, List[str]]
    confidence: float
    response: str
    action_data: Optional[Dict] = None

class FaultKnowledgeBase:
    """Knowledge base for fault types and explanations"""
    
    FAULT_DEFINITIONS = {
        'lg': {
            'full_name': 'Line-to-Ground',
            'description': 'A fault between one phase conductor and ground',
            'characteristics': 'Single phase voltage drops to zero or near zero',
            'common_causes': 'Insulation failure, tree contact, equipment failure'
        },
        'llg': {
            'full_name': 'Line-to-Line-to-Ground',
            'description': 'A fault involving two phases and ground',
            'characteristics': 'Two phase voltages affected, unbalanced system',
            'common_causes': 'Multiple insulation failures, severe weather conditions'
        },
        'll': {
            'full_name': 'Line-to-Line',
            'description': 'A fault between two phase conductors',
            'characteristics': 'Two phases short-circuited, third phase unaffected',
            'common_causes': 'Conductor contact, insulation breakdown'
        },
        'lll': {
            'full_name': 'Three-Phase',
            'description': 'A fault involving all three phases',
            'characteristics': 'Balanced three-phase fault, most severe type',
            'common_causes': 'Equipment failure, major insulation breakdown'
        },
        'lllg': {
            'full_name': 'Three-Phase-to-Ground',
            'description': 'A fault involving all three phases and ground',
            'characteristics': 'Complete system failure, maximum fault current',
            'common_causes': 'Major equipment failure, catastrophic events'
        }
    }

class NaturalLanguageProcessor:
    """Enhanced NLP processor for power system queries"""
    
    def __init__(self):
        self.knowledge_base = FaultKnowledgeBase()
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns()
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build regex patterns for intent recognition"""
        return {
            QueryIntent.SHOW_FAULTS: [
                r'\b(show|display|list|view|get)\b.*\bfaults?\b',
                r'\bfaults?\b.*\b(show|display|list|view)\b',
                r'\b(plot|graph|chart)\b.*\bfaults?\b',
                r'\bgive me.*\bfaults?\b'
            ],
            QueryIntent.FILTER_FAULTS: [
                r'\b(filter|find|search|look for)\b.*\bfaults?\b',
                r'\bfaults?\b.*\b(containing|involving|with|of type|that are)\b',
                r'\b(only|just)\b.*\bfaults?\b',
                r'\bfaults?\b.*\bphase [abc]\b'
            ],
            QueryIntent.EXPLAIN_FAULT: [
                r'\b(what|explain|define|describe)\b.*\b(is|are)\b.*\bfault\b',
                r'\b(tell me about|describe)\b.*\bfault\b',
                r'\bfault\b.*\b(definition|explanation|meaning)\b',
                r'\b(what does|meaning of)\b.*\bfault\b'
            ],
            QueryIntent.COUNT_FAULTS: [
                r'\b(how many|count|number of|total)\b.*\bfaults?\b',
                r'\bfaults?\b.*\b(count|total|number)\b',
                r'\btotal.*\bfaults?\b'
            ],
            QueryIntent.ANALYZE_SIGNAL: [
                r'\b(analyze|examine|check|show|plot)\b.*\b(signal|voltage|current)\b',
                r'\b(signal|voltage|current)\b.*\b(analysis|plot|graph|chart)\b'
            ],
            QueryIntent.HELP: [
                r'\b(help|how|what can|commands|options)\b',
                r'^\?+$'
            ]
        }
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build enhanced patterns for entity extraction"""
        return {
            'fault_types': [
                r'\b(lg|l-g|line to ground|line-to-ground|ground fault)\b',
                r'\b(llg|l-l-g|line to line to ground|line-to-line-to-ground|double line ground)\b',
                r'\b(ll|l-l|line to line|line-to-line|phase to phase)\b',
                r'\b(lll|l-l-l|three phase|three-phase|3-phase|3 phase)\b',
                r'\b(lllg|l-l-l-g|three phase to ground|three-phase-to-ground|3-phase-ground)\b'
            ],
            'phases': [
                r'\bphase [abc]\b',
                r'\b[abc] phase\b',
                r'\bv[abc]\b',
                r'\bi[abc]\b'
            ],
            'signals': [
                r'\b(voltage|voltages|v)\b',
                r'\b(current|currents|i)\b',
                r'\b(signal|signals)\b'
            ]
        }
    
    def process_query(self, query: str) -> QueryResult:
        """Process natural language query and return structured result"""
        query_lower = query.lower().strip()
        
        # Extract intent
        intent = self._extract_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(intent, entities, query_lower)
        
        # Generate response
        response, action_data = self._generate_response(intent, entities, query_lower)
        
        return QueryResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            response=response,
            action_data=action_data
        )
    
    def _extract_intent(self, query: str) -> QueryIntent:
        """Extract intent from query using pattern matching"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score
        
        # Return intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent if intent_scores[best_intent] > 0 else QueryIntent.UNKNOWN
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, query, re.IGNORECASE)
                if found:
                    matches.extend(found)
            
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _calculate_confidence(self, intent: QueryIntent, entities: Dict, query: str) -> float:
        """Calculate confidence score for the interpretation"""
        base_confidence = 0.7 if intent != QueryIntent.UNKNOWN else 0.2
        
        # Boost confidence based on entity matches
        entity_boost = min(0.2, len(entities) * 0.1)
        
        # Boost confidence for clear keywords
        keyword_boost = 0
        power_keywords = ['fault', 'voltage', 'current', 'phase', 'power', 'system', 'lg', 'll', 'lll']
        for keyword in power_keywords:
            if keyword in query.lower():
                keyword_boost += 0.02
        
        return min(1.0, base_confidence + entity_boost + keyword_boost)
    
    def _generate_response(self, intent: QueryIntent, entities: Dict, query: str) -> Tuple[str, Optional[Dict]]:
        """Generate response based on intent and entities"""
        
        if intent == QueryIntent.EXPLAIN_FAULT:
            return self._handle_explain_fault(entities)
        
        elif intent == QueryIntent.SHOW_FAULTS:
            return self._handle_show_faults(entities)
        
        elif intent == QueryIntent.FILTER_FAULTS:
            return self._handle_filter_faults(entities)
        
        elif intent == QueryIntent.COUNT_FAULTS:
            return self._handle_count_faults(entities)
        
        elif intent == QueryIntent.ANALYZE_SIGNAL:
            return self._handle_analyze_signal(entities)
        
        elif intent == QueryIntent.HELP:
            return self._handle_help()
        
        else:
            return self._handle_unknown(query)
    
    def _handle_explain_fault(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        """Handle fault explanation queries"""
        fault_types = entities.get('fault_types', [])
        
        if fault_types:
            fault_type = self._normalize_fault_type(fault_types[0])
            if fault_type in self.knowledge_base.FAULT_DEFINITIONS:
                fault_info = self.knowledge_base.FAULT_DEFINITIONS[fault_type]
                response = f"""**{fault_info['full_name']} ({fault_type.upper()}) Fault:**

📋 **Definition:** {fault_info['description']}

⚡ **Characteristics:** {fault_info['characteristics']}

🔧 **Common Causes:** {fault_info['common_causes']}"""
                return response, {'fault_type': fault_type}
            else:
                return f"I don't have information about '{fault_types[0]}' fault type.", None
        else:
            return "Please specify which fault type you'd like to learn about (e.g., LG, LLG, LL, LLL, LLLG).", None
    
    def _handle_show_faults(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        """Handle show faults queries"""
        action_data = {
            'action': 'show_faults',
            'filters': {}
        }
        
        # Add fault type filters
        if 'fault_types' in entities:
            fault_types = [self._normalize_fault_type(ft) for ft in entities['fault_types']]
            action_data['filters']['fault_types'] = fault_types
        
        # Add phase filters
        if 'phases' in entities:
            phases = [self._normalize_phase(phase) for phase in entities['phases']]
            action_data['filters']['phases'] = phases
        
        response = "I'll show the requested faults"
        if action_data['filters']:
            filter_desc = []
            if 'fault_types' in action_data['filters']:
                filter_desc.append(f"Type: {', '.join(action_data['filters']['fault_types']).upper()}")
            if 'phases' in action_data['filters']:
                filter_desc.append(f"Phases: {', '.join(action_data['filters']['phases']).upper()}")
            response += f" - Filters: {'; '.join(filter_desc)}"
        
        return response, action_data
    
    def _handle_filter_faults(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        """Handle filter faults queries"""
        return self._handle_show_faults(entities)
    
    def _handle_count_faults(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        """Handle count faults queries"""
        action_data = {
            'action': 'count_faults',
            'filters': {}
        }
        
        if 'fault_types' in entities:
            fault_types = [self._normalize_fault_type(ft) for ft in entities['fault_types']]
            action_data['filters']['fault_types'] = fault_types
        
        if 'phases' in entities:
            phases = [self._normalize_phase(phase) for phase in entities['phases']]
            action_data['filters']['phases'] = phases
        
        return "I'll count the faults based on your criteria", action_data
    
    def _handle_analyze_signal(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        """Handle signal analysis queries"""
        signals = entities.get('signals', [])
        phases = entities.get('phases', [])
        
        action_data = {
            'action': 'analyze_signal',
            'signals': signals,
            'phases': [self._normalize_phase(phase) for phase in phases]
        }
        
        return "I'll analyze the requested signals", action_data
    
    def _handle_help(self) -> Tuple[str, Optional[Dict]]:
        """Handle help queries"""
        response = """🤖 **Natural Language Query Help**

**Fault Information:**
- "What is an LG fault?"
- "Explain LLG faults"
- "Tell me about three-phase faults"

**Show/Filter Faults:**
- "Show all LG faults"
- "Display faults involving phase B"
- "List three-phase faults"

**Count Faults:**
- "How many LG faults are there?"
- "Count faults with phase A"

**Signal Analysis:**
- "Analyze voltage signals"
- "Show current in phase B"

**Supported:** LG, LLG, LL, LLL, LLLG | Phases A, B, C"""
        return response, None
    
    def _handle_unknown(self, query: str) -> Tuple[str, Optional[Dict]]:
        """Handle unknown queries"""
        return f"I'm not sure how to interpret '{query}'. Type 'help' to see available commands.", None
    
    def _normalize_fault_type(self, fault_type: str) -> str:
        """Normalize fault type to standard format"""
        fault_type = fault_type.lower().replace('-', '').replace(' ', '')
        mapping = {
            'linetoground': 'lg',
            'groundfault': 'lg',
            'linetolinetoground': 'llg',
            'doublelineground': 'llg',
            'linetoline': 'll',
            'phasetophase': 'll',
            'threephase': 'lll',
            '3phase': 'lll',
            'threephasetoground': 'lllg',
            '3phaseground': 'lllg'
        }
        return mapping.get(fault_type, fault_type)
    
    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase to standard format"""
        phase = phase.lower()
        if 'a' in phase:
            return 'a'
        elif 'b' in phase:
            return 'b'
        elif 'c' in phase:
            return 'c'
        return phase

class QueryInterface:
    """Main interface for natural language queries"""
    
    def __init__(self):
        self.processor = NaturalLanguageProcessor()
        self.query_history = []
    
    def process_query(self, query: str, session_state) -> QueryResult:
        """Process query and update session state"""
        result = self.processor.process_query(query)
        
        # Add to history
        self.query_history.append({
            'query': query,
            'result': result,
            'timestamp': pd.Timestamp.now()
        })
        
        # Execute actions based on result
        if result.action_data:
            self._execute_action(result.action_data, session_state)
        
        return result
    
    def _execute_action(self, action_data: Dict, session_state):
        """Execute actions based on query result"""
        action = action_data.get('action')
        
        if action in ['show_faults', 'filter_faults', 'count_faults']:
            # Initialize if not exists
            if not hasattr(session_state, 'nlp_filters'):
                session_state.nlp_filters = {}
            if not hasattr(session_state, 'nlp_action'):
                session_state.nlp_action = None
                
            session_state.nlp_filters = action_data.get('filters', {})
            session_state.nlp_action = action
        
        elif action == 'analyze_signal':
            session_state.nlp_signal_request = action_data
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input"""
        if len(partial_query) < 2:
            return []
            
        suggestions = []
        partial_lower = partial_query.lower()
        
        templates = [
            "Show all LG faults",
            "What is an LLG fault?",
            "Count faults involving phase A",
            "Display three-phase faults",
            "Analyze voltage signals",
            "Show faults with phase B",
            "How many faults are there?",
            "Explain line-to-ground faults",
            "Help me",
            "List all fault types"
        ]
        
        for template in templates:
            if (partial_lower in template.lower() or 
                template.lower().startswith(partial_lower) or
                any(word.startswith(partial_lower) for word in template.lower().split())):
                suggestions.append(template)
        
        return suggestions[:5]

def apply_nlp_filters(results_df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply NLP-based filters to fault results"""
    if results_df.empty or not filters:
        return results_df
    
    filtered_df = results_df.copy()
    
    # Filter by fault types
    if 'fault_types' in filters and filters['fault_types']:
        fault_pattern = '|'.join([f'^{ft}$' for ft in filters['fault_types']])
        filtered_df = filtered_df[
            filtered_df['fault_type'].str.contains(fault_pattern, case=False, na=False, regex=True)
        ]
    
    return filtered_df

def create_nlp_response_display(result: QueryResult):
    """Create enhanced Streamlit display for NLP query results"""
    # Show confidence if low
    if result.confidence < 0.6:
        st.warning(f"⚠️ Low confidence ({result.confidence:.2f}) - please be more specific")
    
    # Display response based on intent
    if result.intent == QueryIntent.EXPLAIN_FAULT:
        st.markdown("### 📚 Fault Explanation")
        st.markdown(result.response)
    
    elif result.intent in [QueryIntent.SHOW_FAULTS, QueryIntent.FILTER_FAULTS]:
        st.markdown("### 🔍 Query Processing")
        st.success(result.response)
    
    elif result.intent == QueryIntent.COUNT_FAULTS:
        st.markdown("### 📊 Fault Counting")
        st.success(result.response)
    
    elif result.intent == QueryIntent.HELP:
        st.markdown("### 🆘 Help")
        st.markdown(result.response)
    
    else:
        st.markdown("### 🤖 Response")
        st.info(result.response)

# Tab4 Implementation
def create_tab4_interface():
    """Create the complete Tab4 interface"""
    
    # Initialize query interface in session state if not exists
    if 'query_interface' not in st.session_state:
        st.session_state.query_interface = QueryInterface()
    
    st.markdown("## 🤖 Smart Query Interface")
    st.markdown("Ask questions about your power system data in natural language!")
    
    # Query input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use session state to maintain input value
        if 'query_input_value' not in st.session_state:
            st.session_state.query_input_value = ""
            
        query_input = st.text_input(
            "💬 Ask a question:",
            value=st.session_state.query_input_value,
            placeholder="e.g., 'Show all LG faults' or 'What is an LLG fault?'",
            help="Type your question in natural language",
            key="main_query_input"
        )
    
    with col2:
        process_query = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Query suggestions
    if query_input:
        suggestions = st.session_state.query_interface.get_suggestions(query_input)
        if suggestions:
            st.markdown("**💡 Suggestions:**")
            cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i]:
                    if st.button(f"💭 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                        st.session_state.query_input_value = suggestion
                        st.rerun()
    
    # Process query
    if process_query and query_input:
        with st.spinner("Processing your query..."):
            result = st.session_state.query_interface.process_query(query_input, st.session_state)
            
            # Display result
            create_nlp_response_display(result)
            
            # Handle data queries if analysis is complete
            if (result.intent in [QueryIntent.SHOW_FAULTS, QueryIntent.FILTER_FAULTS, QueryIntent.COUNT_FAULTS] 
                and hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete):
                
                results = st.session_state.results
                if results and 'classification_results' in results and results['classification_results']:
                    classification_df = pd.DataFrame(results['classification_results'])
                    successful_df = classification_df[classification_df['fault_type'].notna()].copy()
                    
                    if not successful_df.empty:
                        # Apply NLP filters
                        filters = getattr(st.session_state, 'nlp_filters', {})
                        filtered_df = apply_nlp_filters(successful_df, filters)
                        
                        if result.intent == QueryIntent.COUNT_FAULTS:
                            st.markdown("### 📊 Count Results")
                            if filtered_df.empty:
                                st.info("No faults found matching your criteria")
                            else:
                                st.success(f"Found **{len(filtered_df)}** faults matching your criteria")
                                
                                # Show breakdown
                                if len(filtered_df) > 0:
                                    fault_counts = filtered_df['fault_type'].value_counts()
                                    st.markdown("**Breakdown by fault type:**")
                                    for fault_type, count in fault_counts.items():
                                        st.write(f"- **{fault_type.upper()}**: {count}")
                        
                        else:  # SHOW_FAULTS or FILTER_FAULTS
                            st.markdown("### 🎯 Filtered Results")
                            if filtered_df.empty:
                                st.info("No faults found matching your criteria")
                            else:
                                # Display table
                                display_cols = ['interval', 'start_time', 'end_time', 'duration', 'fault_type', 'confidence']
                                display_df = filtered_df[display_cols].copy()
                                display_df = display_df.round({'start_time': 3, 'end_time': 3, 'duration': 3, 'confidence': 3})
                                
                                st.dataframe(
                                    display_df,
                                    column_config={
                                        "interval": "Fault #",
                                        "start_time": "Start (s)",
                                        "end_time": "End (s)",
                                        "duration": "Duration (s)",
                                        "fault_type": "Type",
                                        "confidence": st.column_config.ProgressColumn(
                                            "Confidence",
                                            min_value=0,
                                            max_value=1,
                                        ),
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                st.success(f"Showing {len(filtered_df)} of {len(successful_df)} total faults")
                    else:
                        st.warning("No successful fault classifications found")
                else:
                    st.warning("⚠️ No analysis results available. Please run the analysis first in the Classification tab.")
    
    # Query history
    if st.session_state.query_interface.query_history:
        st.markdown("### 📝 Recent Queries")
        
        with st.expander("View Query History"):
            for i, query_record in enumerate(reversed(st.session_state.query_interface.query_history[-5:])):
                st.markdown(f"**{i+1}.** {query_record['query']}")
                st.caption(f"Response: {query_record['result'].response[:100]}{'...' if len(query_record['result'].response) > 100 else ''}")
                if query_record['result'].confidence < 0.6:
                    st.caption(f"⚠️ Low confidence: {query_record['result'].confidence:.2f}")
                st.divider()
    
    # Help section
    with st.expander("🆘 Query Examples & Help"):
        st.markdown("""
        **Example Queries:**
        
        **🔍 Fault Information:**
        - "What is an LG fault?"
        - "Explain three-phase faults"
        
        **📊 Show/Filter Faults:**
        - "Show all LG faults"
        - "List three-phase faults"
        - "Display LLG faults"
        
        **🔢 Count Faults:**
        - "How many LG faults?"
        - "Count three-phase faults"
        
        **Supported Types:** LG, LLG, LL, LLL, LLLG
        
        **Tips:**
        - Be specific about fault types
        - Use natural language
        - Run analysis first for data queries
        """)

    st.markdown("---")
    st.markdown("*Natural Language Processing for Power System Analysis*")