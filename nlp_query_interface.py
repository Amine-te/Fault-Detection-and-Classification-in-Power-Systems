import spacy
import streamlit as st
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from spacy.matcher import Matcher
from spacy.util import filter_spans

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

class SpacyNLPProcessor:
    """Enhanced NLP processor using spaCy for power system queries"""
    
    def __init__(self):
        self.knowledge_base = FaultKnowledgeBase()
        self.nlp = self._load_spacy_model()
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        
    @st.cache_resource
    def _load_spacy_model(_self):
        """Load spaCy model with caching"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error("""
            **spaCy model not found!** 
            
            Please install it using:
            ```bash
            python -m spacy download en_core_web_sm
            ```
            """)
            return None
    
    def _setup_patterns(self):
        """Setup spaCy patterns for intent and entity recognition"""
        if not self.nlp:
            return
            
        # Intent patterns
        intent_patterns = {
            "SHOW_INTENT": [
                [{"LOWER": {"IN": ["show", "display", "list", "view", "get"]}},
                 {"LOWER": "faults", "OP": "?"}],
                [{"LOWER": {"IN": ["plot", "graph", "chart"]}},
                 {"LOWER": "faults", "OP": "?"}]
            ],
            "EXPLAIN_INTENT": [
                [{"LOWER": {"IN": ["what", "explain", "define", "describe"]}},
                 {"IS_ALPHA": True, "OP": "*"},
                 {"LOWER": "fault"}],
                [{"LOWER": {"IN": ["tell", "describe"]}},
                 {"LOWER": "me", "OP": "?"},
                 {"LOWER": "about"}]
            ],
            "COUNT_INTENT": [
                [{"LOWER": {"IN": ["how", "count", "number", "total"]}},
                 {"LOWER": {"IN": ["many", "of"]}, "OP": "?"},
                 {"LOWER": "faults", "OP": "?"}]
            ],
            "FILTER_INTENT": [
                [{"LOWER": {"IN": ["filter", "find", "search", "look"]}},
                 {"LOWER": {"IN": ["for", "faults"]}, "OP": "?"}],
                [{"LOWER": "faults"},
                 {"LOWER": {"IN": ["with", "containing", "involving"]}}]
            ],
            "ANALYZE_INTENT": [
                [{"LOWER": {"IN": ["analyze", "examine", "check"]}},
                 {"LOWER": {"IN": ["signal", "voltage", "current"]}, "OP": "?"}]
            ],
            "HELP_INTENT": [
                [{"LOWER": {"IN": ["help", "how", "commands", "options"]}}]
            ]
        }
        
        # Entity patterns
        entity_patterns = {
            "FAULT_TYPE": [
                [{"LOWER": {"IN": ["lg", "l-g"]}}],
                [{"LOWER": "line"}, {"LOWER": "to"}, {"LOWER": "ground"}],
                [{"LOWER": {"IN": ["llg", "l-l-g"]}}],
                [{"LOWER": "line"}, {"LOWER": "to"}, {"LOWER": "line"}, {"LOWER": "to"}, {"LOWER": "ground"}],
                [{"LOWER": {"IN": ["ll", "l-l"]}}],
                [{"LOWER": "line"}, {"LOWER": "to"}, {"LOWER": "line"}],
                [{"LOWER": {"IN": ["lll", "l-l-l", "three-phase", "3-phase"]}}],
                [{"LOWER": "three"}, {"LOWER": "phase"}],
                [{"LOWER": {"IN": ["lllg", "l-l-l-g"]}}],
                [{"LOWER": "three"}, {"LOWER": "phase"}, {"LOWER": "to"}, {"LOWER": "ground"}]
            ],
            "PHASE": [
                [{"LOWER": "phase"}, {"LOWER": {"IN": ["a", "b", "c"]}}],
                [{"LOWER": {"IN": ["a", "b", "c"]}}, {"LOWER": "phase"}],
                [{"LOWER": {"IN": ["va", "vb", "vc", "ia", "ib", "ic"]}}]
            ],
            "SIGNAL": [
                [{"LOWER": {"IN": ["voltage", "voltages", "current", "currents", "signal", "signals"]}}]
            ]
        }
        
        # Add patterns to matcher
        for intent, patterns in intent_patterns.items():
            self.matcher.add(intent, patterns)
            
        for entity, patterns in entity_patterns.items():
            self.matcher.add(entity, patterns)
    
    def process_query(self, query: str) -> QueryResult:
        """Process natural language query using spaCy"""
        if not self.nlp:
            return QueryResult(
                intent=QueryIntent.UNKNOWN,
                entities={},
                confidence=0.0,
                response="spaCy model not available. Please install en_core_web_sm.",
                action_data=None
            )
        
        doc = self.nlp(query)
        
        # Extract intent and entities using matcher
        matches = self.matcher(doc)
        
        # Process matches
        intent_matches = []
        entity_matches = {}
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            if "INTENT" in label:
                intent_matches.append(label)
            else:
                if label not in entity_matches:
                    entity_matches[label] = []
                entity_matches[label].append(span.text.lower())
        
        # Determine intent
        intent = self._determine_intent(intent_matches, doc)
        
        # Extract entities with semantic analysis
        entities = self._extract_entities_semantic(doc, entity_matches)
        
        # Calculate confidence
        confidence = self._calculate_confidence_semantic(intent, entities, doc)
        
        # Generate response
        response, action_data = self._generate_response(intent, entities, query)
        
        return QueryResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            response=response,
            action_data=action_data
        )
    
    def _determine_intent(self, intent_matches: List[str], doc) -> QueryIntent:
        """Determine intent using spaCy analysis"""
        # Priority mapping
        intent_priority = {
            "EXPLAIN_INTENT": QueryIntent.EXPLAIN_FAULT,
            "COUNT_INTENT": QueryIntent.COUNT_FAULTS,
            "FILTER_INTENT": QueryIntent.FILTER_FAULTS,
            "SHOW_INTENT": QueryIntent.SHOW_FAULTS,
            "ANALYZE_INTENT": QueryIntent.ANALYZE_SIGNAL,
            "HELP_INTENT": QueryIntent.HELP
        }
        
        # Use highest priority match
        for intent_match in intent_matches:
            if intent_match in intent_priority:
                return intent_priority[intent_match]
        
        # Fallback: analyze verbs and keywords
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        keywords = [token.text.lower() for token in doc]
        
        if any(v in ["explain", "define", "describe"] for v in verbs):
            return QueryIntent.EXPLAIN_FAULT
        elif any(v in ["count", "number"] for v in verbs) or "how many" in doc.text.lower():
            return QueryIntent.COUNT_FAULTS
        elif any(v in ["show", "display", "list"] for v in verbs):
            return QueryIntent.SHOW_FAULTS
        elif any(v in ["analyze", "examine"] for v in verbs):
            return QueryIntent.ANALYZE_SIGNAL
        elif "help" in keywords:
            return QueryIntent.HELP
        else:
            return QueryIntent.UNKNOWN
    
    def _extract_entities_semantic(self, doc, entity_matches: Dict) -> Dict[str, List[str]]:
        """Extract entities using semantic analysis"""
        entities = {}
        
        # Process matched entities
        for entity_type, matches in entity_matches.items():
            normalized_matches = []
            for match in matches:
                if entity_type == "FAULT_TYPE":
                    normalized_matches.append(self._normalize_fault_type(match))
                elif entity_type == "PHASE":
                    normalized_matches.append(self._normalize_phase(match))
                elif entity_type == "SIGNAL":
                    normalized_matches.append(match)
            
            if normalized_matches:
                key = entity_type.lower() + "s"
                entities[key] = list(set(normalized_matches))  # Remove duplicates
        
        # Additional semantic extraction using NER and dependency parsing
        for ent in doc.ents:
            if ent.label_ in ["CARDINAL", "ORDINAL"] and "phase" in doc.text.lower():
                # Extract phase numbers
                if "phases" not in entities:
                    entities["phases"] = []
                if ent.text.lower() in ["a", "b", "c", "1", "2", "3"]:
                    phase_map = {"1": "a", "2": "b", "3": "c"}
                    phase = phase_map.get(ent.text.lower(), ent.text.lower())
                    if phase not in entities["phases"]:
                        entities["phases"].append(phase)
        
        return entities
    
    def _calculate_confidence_semantic(self, intent: QueryIntent, entities: Dict, doc) -> float:
        """Calculate confidence using semantic features"""
        base_confidence = 0.8 if intent != QueryIntent.UNKNOWN else 0.3
        
        # Boost for clear power system terminology
        power_terms = ["fault", "voltage", "current", "phase", "power", "system", "ground"]
        term_score = sum(1 for token in doc if token.text.lower() in power_terms)
        term_boost = min(0.15, term_score * 0.03)
        
        # Boost for entity matches
        entity_boost = min(0.15, len(entities) * 0.05)
        
        # Penalty for very short queries
        length_penalty = 0.1 if len(doc) < 3 else 0
        
        # Boost for grammatically correct sentences
        grammar_boost = 0.05 if any(token.dep_ == "ROOT" for token in doc) else 0
        
        confidence = base_confidence + term_boost + entity_boost + grammar_boost - length_penalty
        return min(1.0, max(0.0, confidence))
    
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
            fault_type = fault_types[0]
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
        action_data = {'action': 'show_faults', 'filters': {}}
        
        if 'fault_types' in entities:
            action_data['filters']['fault_types'] = entities['fault_types']
        if 'phases' in entities:
            action_data['filters']['phases'] = entities['phases']
        
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
        return self._handle_show_faults(entities)
    
    def _handle_count_faults(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        action_data = {'action': 'count_faults', 'filters': {}}
        
        if 'fault_types' in entities:
            action_data['filters']['fault_types'] = entities['fault_types']
        if 'phases' in entities:
            action_data['filters']['phases'] = entities['phases']
        
        return "I'll count the faults based on your criteria", action_data
    
    def _handle_analyze_signal(self, entities: Dict) -> Tuple[str, Optional[Dict]]:
        signals = entities.get('signals', [])
        phases = entities.get('phases', [])
        
        action_data = {
            'action': 'analyze_signal',
            'signals': signals,
            'phases': phases
        }
        
        return "I'll analyze the requested signals", action_data
    
    def _handle_help(self) -> Tuple[str, Optional[Dict]]:
        response = """🤖 **spaCy-Enhanced Query Help**

**🔍 Fault Information:**
- "What is an LG fault?"
- "Explain line-to-ground faults"
- "Tell me about three-phase faults"

**📊 Show/Filter Faults:**
- "Show all LG faults"
- "Display faults involving phase B"
- "List three-phase faults"

**🔢 Count Faults:**
- "How many LG faults are there?"
- "Count faults with phase A"

**📈 Signal Analysis:**
- "Analyze voltage signals"
- "Show current in phase B"

**✨ Enhanced Features:**
- Better context understanding
- Semantic entity recognition
- Improved confidence scoring
- Natural language flexibility

**Supported:** LG, LLG, LL, LLL, LLLG | Phases A, B, C"""
        return response, None
    
    def _handle_unknown(self, query: str) -> Tuple[str, Optional[Dict]]:
        return f"I'm not sure how to interpret '{query}'. Try asking about fault types, analysis, or type 'help'.", None
    
    def _normalize_fault_type(self, fault_type: str) -> str:
        """Normalize fault type to standard format"""
        fault_type = fault_type.lower().replace('-', '').replace(' ', '')
        mapping = {
            'linetoground': 'lg', 'groundfault': 'lg',
            'linetolinetoground': 'llg', 'doublelineground': 'llg',
            'linetoline': 'll', 'phasetophase': 'll',
            'threephase': 'lll', '3phase': 'lll',
            'threephasetoground': 'lllg', '3phaseground': 'lllg'
        }
        return mapping.get(fault_type, fault_type)
    
    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase to standard format"""
        phase = phase.lower()
        if 'a' in phase: return 'a'
        elif 'b' in phase: return 'b'
        elif 'c' in phase: return 'c'
        return phase

class QueryInterface:
    """Main interface for spaCy-based natural language queries"""
    
    def __init__(self):
        self.processor = SpacyNLPProcessor()
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
            if not hasattr(session_state, 'nlp_filters'):
                session_state.nlp_filters = {}
            if not hasattr(session_state, 'nlp_action'):
                session_state.nlp_action = None
                
            session_state.nlp_filters = action_data.get('filters', {})
            session_state.nlp_action = action
        
        elif action == 'analyze_signal':
            session_state.nlp_signal_request = action_data
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get intelligent query suggestions"""
        if len(partial_query) < 2:
            return []
            
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Smart templates based on input
        templates = {
            'show': ["Show all LG faults", "Show three-phase faults", "Show faults with phase A"],
            'what': ["What is an LG fault?", "What are LLG faults?", "What is a three-phase fault?"],
            'count': ["Count LG faults", "Count all faults", "How many faults are there?"],
            'analyze': ["Analyze voltage signals", "Analyze current", "Analyze phase A voltage"],
            'explain': ["Explain LG faults", "Explain three-phase faults"],
            'help': ["Help me", "Show commands", "What can you do?"]
        }
        
        # Find matching templates
        for key, template_list in templates.items():
            if partial_lower.startswith(key) or key in partial_lower:
                suggestions.extend(template_list)
        
        # General suggestions if no specific match
        if not suggestions:
            suggestions = [
                "Show all LG faults",
                "What is an LLG fault?",
                "Count three-phase faults",
                "Analyze voltage signals",
                "Help me"
            ]
        
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
    # Show confidence with color coding
    if result.confidence >= 0.8:
        st.success(f"✅ High confidence ({result.confidence:.2f})")
    elif result.confidence >= 0.6:
        st.info(f"ℹ️ Medium confidence ({result.confidence:.2f})")
    else:
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
    
    # Show extracted entities
    if result.entities:
        with st.expander("🔍 Extracted Entities"):
            for entity_type, entities in result.entities.items():
                st.write(f"**{entity_type.title()}:** {', '.join(entities)}")

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
            placeholder="e.g., 'Show all LG faults' or 'What is an LLG fault?' or 'Analyze voltage signals'",
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
            
            # Handle signal analysis specifically
            if (result.intent == QueryIntent.ANALYZE_SIGNAL and 
                hasattr(st.session_state, 'nlp_signal_request')):
                
                # Check if data is available (more flexible check)
                data_available = False
                data_source = None
                
                # Check multiple possible data sources
                if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                    data_available = True
                    data_source = st.session_state.data
                    st.info("✅ Using uploaded data for signal analysis")
                elif (hasattr(st.session_state, 'processed_data') and 
                    st.session_state.processed_data is not None):
                    data_available = True
                    data_source = st.session_state.processed_data
                    st.info("✅ Using processed data for signal analysis")
                elif (hasattr(st.session_state, 'analysis_complete') and 
                    st.session_state.analysis_complete and 
                    hasattr(st.session_state, 'data')):
                    data_available = True
                    data_source = st.session_state.data
                    st.info("✅ Using analyzed data for signal analysis")
                
                if data_available and data_source is not None:
                    st.markdown("### 📈 Signal Analysis")
                    
                    # Get signal analysis parameters
                    signal_request = st.session_state.nlp_signal_request
                    signals = signal_request.get('signals', ['voltage'])
                    phases = signal_request.get('phases', ['a', 'b', 'c'])
                    
                    # Display available signals
                    data = data_source
                    available_signals = list(data.columns)
                    
                    st.info(f"📊 Found {len(available_signals)} available signals in the dataset")
                    
                    # Create signal selection based on NLP request
                    signal_columns = []
                    
                    # Map NLP requests to actual column names (improved logic)
                    for signal_type in signals:
                        if signal_type in ['voltage', 'voltages']:
                            # Look for voltage columns
                            voltage_cols = []
                            for col in available_signals:
                                col_upper = col.upper()
                                if ('V' in col_upper and ('_A' in col_upper or '_B' in col_upper or '_C' in col_upper)) or \
                                any(f'V{phase.upper()}' in col_upper or f'{phase.upper()}V' in col_upper for phase in phases) or \
                                ('VOLTAGE' in col_upper):
                                    voltage_cols.append(col)
                            
                            if not voltage_cols:  # Fallback to any column with V
                                voltage_cols = [col for col in available_signals if 'V' in col.upper()]
                            
                            signal_columns.extend(voltage_cols)
                        
                        elif signal_type in ['current', 'currents']:
                            # Look for current columns
                            current_cols = []
                            for col in available_signals:
                                col_upper = col.upper()
                                if ('I' in col_upper and ('_A' in col_upper or '_B' in col_upper or '_C' in col_upper)) or \
                                any(f'I{phase.upper()}' in col_upper or f'{phase.upper()}I' in col_upper for phase in phases) or \
                                ('CURRENT' in col_upper):
                                    current_cols.append(col)
                            
                            if not current_cols:  # Fallback to any column with I
                                current_cols = [col for col in available_signals if 'I' in col.upper()]
                            
                            signal_columns.extend(current_cols)
                        
                        elif signal_type in ['signal', 'signals']:
                            # General signal request - show first few numeric columns
                            numeric_cols = []
                            for col in available_signals:
                                try:
                                    if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                        numeric_cols.append(col)
                                except:
                                    pass
                            signal_columns.extend(numeric_cols[:6])  # First 6 numeric columns
                    
                    # If no specific signals found, default to first few numeric columns
                    if not signal_columns:
                        st.warning("No specific signal columns found. Showing first available numeric columns.")
                        numeric_cols = []
                        for col in available_signals:
                            try:
                                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                    numeric_cols.append(col)
                            except:
                                pass
                        signal_columns = numeric_cols[:6]
                    
                    # Remove duplicates and limit
                    signal_columns = list(dict.fromkeys(signal_columns))[:6]  # Max 6 signals
                    
                    if signal_columns:
                        st.success(f"📈 Analyzing signals: {', '.join(signal_columns)}")
                        
                        # Show available vs selected
                        with st.expander("🔍 Signal Selection Details"):
                            st.write("**Available signals:**", ", ".join(available_signals[:20]))
                            if len(available_signals) > 20:
                                st.write(f"... and {len(available_signals) - 20} more")
                            st.write("**Selected for analysis:**", ", ".join(signal_columns))
                        
                        # Create time series plot
                        try:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            # Determine number of subplots
                            n_signals = len(signal_columns)
                            n_rows = min(n_signals, 3)  # Max 3 rows
                            n_cols = 1 if n_signals <= 3 else 2
                            
                            if n_signals == 1:
                                n_rows, n_cols = 1, 1
                            elif n_signals == 2:
                                n_rows, n_cols = 2, 1
                            else:
                                n_rows = (n_signals + 1) // 2
                                n_cols = 2
                            
                            fig = make_subplots(
                                rows=n_rows, 
                                cols=n_cols,
                                subplot_titles=signal_columns,
                                vertical_spacing=0.08,
                                horizontal_spacing=0.05
                            )
                            
                            # Add traces
                            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                            
                            for i, signal in enumerate(signal_columns):
                                row = (i // n_cols) + 1
                                col = (i % n_cols) + 1
                                
                                # Create time index if not available
                                if data.index.name is None and 'time' not in data.columns:
                                    time_vals = range(len(data))
                                    x_title = "Sample Index"
                                else:
                                    time_vals = data.index
                                    x_title = "Time (s)" if 'time' in str(data.index.name).lower() else "Index"
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_vals,
                                        y=data[signal],
                                        name=signal,
                                        line=dict(color=colors[i % len(colors)]),
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                            
                            fig.update_layout(
                                height=200 * n_rows,
                                title_text="Signal Analysis Results",
                                title_x=0.5
                            )
                            
                            # Update axes
                            for i in range(1, n_rows + 1):
                                for j in range(1, n_cols + 1):
                                    fig.update_xaxes(title_text=x_title, row=i, col=j)
                                    fig.update_yaxes(title_text="Amplitude", row=i, col=j)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error creating plot: {str(e)}")
                            st.info("Showing signal data in table format instead:")
                            
                            # Show data table as fallback
                            display_data = data[signal_columns].head(100)  # First 100 rows
                            st.dataframe(display_data, use_container_width=True)
                        
                        # Show signal statistics
                        st.markdown("### 📊 Signal Statistics")
                        try:
                            stats_df = data[signal_columns].describe().round(4)
                            st.dataframe(stats_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error calculating statistics: {str(e)}")
                        
                        # Show fault intervals if available
                        if (hasattr(st.session_state, 'results') and 
                            st.session_state.results and 
                            'classification_results' in st.session_state.results):
                            
                            classification_results = st.session_state.results['classification_results']
                            if classification_results:
                                st.markdown("### ⚡ Fault Intervals in Signal")
                                fault_df = pd.DataFrame(classification_results)
                                successful_faults = fault_df[fault_df['fault_type'].notna()]
                                
                                if not successful_faults.empty:
                                    st.dataframe(
                                        successful_faults[['interval', 'start_time', 'end_time', 'fault_type', 'confidence']].round(3),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.info("No fault intervals detected in the signal")
                    
                    else:
                        st.error("❌ No suitable signal columns found for analysis")
                        
                        # Show debug information
                        with st.expander("🔧 Debug Information"):
                            st.write("**All available columns:**")
                            for i, col in enumerate(available_signals):
                                col_type = str(data[col].dtype) if col in data.columns else "unknown"
                                st.write(f"{i+1}. {col} ({col_type})")
                            
                            st.write("**Requested signals:**", signals)
                            st.write("**Requested phases:**", phases)
                        
                        # Allow manual column selection
                        st.markdown("### 🎛️ Manual Signal Selection")
                        selected_cols = st.multiselect(
                            "Select columns to analyze:",
                            options=available_signals,
                            default=available_signals[:3] if len(available_signals) >= 3 else available_signals,
                            key="manual_signal_selection"
                        )
                        
                        if selected_cols and st.button("📈 Analyze Selected Signals"):
                            # Rerun analysis with manually selected columns
                            st.session_state.nlp_signal_request = {
                                'action': 'analyze_signal',
                                'signals': ['signal'],  # Generic signal type
                                'phases': ['a', 'b', 'c'],
                                'manual_selection': selected_cols
                            }
                            st.rerun()
                
                else:
                    st.warning("⚠️ No data available for signal analysis.")
                    st.info("""
                    **To analyze signals, you need to:**
                    1. Upload data in the **Data Upload** tab
                    2. Ensure your data contains numeric columns
                    3. Then come back here and ask to analyze signals
                    
                    **Available data sources checked:**
                    - st.session_state.data
                    - st.session_state.processed_data
                    - Analysis results data
                    """)
                    
                    # Debug information
                    with st.expander("🔧 Debug Session State"):
                        debug_info = {
                            'has_data': hasattr(st.session_state, 'data'),
                            'has_processed_data': hasattr(st.session_state, 'processed_data'),
                            'has_analysis_complete': hasattr(st.session_state, 'analysis_complete'),
                            'session_state_keys': list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                        }
                        st.json(debug_info)
            
            # Handle data queries if analysis is complete
            elif (result.intent in [QueryIntent.SHOW_FAULTS, QueryIntent.FILTER_FAULTS, QueryIntent.COUNT_FAULTS] 
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
        
        **📈 Signal Analysis:**
        - "Analyze voltage signals"
        - "Show current signals"
        - "Analyze phase A voltage"
        - "Display all signals"
        
        **Supported Types:** LG, LLG, LL, LLL, LLLG
        
        **Tips:**
        - Be specific about fault types
        - Use natural language
        - Run analysis first for data queries
        - Upload data first for signal analysis
        """)

    st.markdown("---")
    st.markdown("*Natural Language Processing for Power System Analysis*")