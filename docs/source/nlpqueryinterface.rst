=======================================
Natural Language Processing Query Interface
=======================================

Overview
--------

The Natural Language Processing (NLP) Query Interface provides an intelligent, conversational way to interact with power system fault analysis data. Users can ask questions in natural language and receive structured responses, making complex power system analysis accessible to both technical and non-technical users.

Features
--------

* **Natural Language Understanding**: Process queries written in everyday English
* **Intent Recognition**: Automatically identify what users want to accomplish
* **Entity Extraction**: Parse technical terms, fault types, and system components
* **Smart Filtering**: Apply complex filters through simple language
* **Knowledge Base**: Built-in explanations of power system concepts
* **Query Suggestions**: Context-aware autocomplete and suggestions
* **Confidence Scoring**: Reliability indicators for query interpretation

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

Simply type your question in the query input field:

.. code-block:: text

   "Show all LG faults"
   "What is a three-phase fault?"
   "How many faults involve phase A?"

The system will automatically:

1. Parse your natural language input
2. Identify your intent (show, explain, count, etc.)
3. Extract relevant entities (fault types, phases, signals)
4. Execute the appropriate action
5. Display results in an intuitive format

Example Queries
---------------

Fault Information Queries
~~~~~~~~~~~~~~~~~~~~~~~~~

Get explanations and definitions for different fault types:

.. code-block:: text

   "What is an LG fault?"
   "Explain LLG faults"
   "Tell me about three-phase faults"
   "Define line-to-ground fault"
   "What does LLLG mean?"

**Response Format**: Detailed explanations including:

* Full fault name and abbreviation
* Technical definition
* Characteristic behaviors
* Common causes

Show/Filter Fault Queries
~~~~~~~~~~~~~~~~~~~~~~~~~

Display and filter fault data:

.. code-block:: text

   "Show all LG faults"
   "Display faults involving phase B"
   "List three-phase faults"
   "Find LLG faults"
   "Get faults with phase A"

**Response Format**: Filtered data table with:

* Fault interval number
* Start and end times
* Duration
* Fault type classification
* Confidence scores

Count Queries
~~~~~~~~~~~~~

Get statistical information about faults:

.. code-block:: text

   "How many LG faults are there?"
   "Count faults with phase A"
   "Total number of three-phase faults"
   "How many faults involve phase B?"

**Response Format**: 

* Total count matching criteria
* Breakdown by fault type
* Summary statistics

Signal Analysis Queries
~~~~~~~~~~~~~~~~~~~~~~~

Request signal analysis and visualization:

.. code-block:: text

   "Analyze voltage signals"
   "Show current in phase B"
   "Plot voltage for phase A"
   "Examine current signals"

Help Queries
~~~~~~~~~~~~

Get assistance and command reference:

.. code-block:: text

   "help"
   "what can you do?"
   "show me commands"
   "?"

Supported Fault Types
---------------------

The system recognizes various fault type formats:

Line-to-Ground (LG)
~~~~~~~~~~~~~~~~~~~

**Recognized Terms**:
* "LG", "L-G"
* "line to ground", "line-to-ground"
* "ground fault"

**Description**: Single-phase fault involving one conductor and ground

Line-to-Line-to-Ground (LLG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recognized Terms**:
* "LLG", "L-L-G"
* "line to line to ground", "line-to-line-to-ground"
* "double line ground"

**Description**: Two-phase fault involving two conductors and ground

Line-to-Line (LL)
~~~~~~~~~~~~~~~~~~

**Recognized Terms**:
* "LL", "L-L"
* "line to line", "line-to-line"
* "phase to phase"

**Description**: Two-phase fault between conductors without ground

Three-Phase (LLL)
~~~~~~~~~~~~~~~~~

**Recognized Terms**:
* "LLL", "L-L-L"
* "three phase", "three-phase"
* "3-phase", "3 phase"

**Description**: Balanced fault involving all three phases

Three-Phase-to-Ground (LLLG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recognized Terms**:
* "LLLG", "L-L-L-G"
* "three phase to ground", "three-phase-to-ground"
* "3-phase-ground"

**Description**: Complete system fault involving all phases and ground

Technical Architecture
----------------------

Core Components
~~~~~~~~~~~~~~~

QueryIntent Enumeration
~~~~~~~~~~~~~~~~~~~~~~~~

Defines the different types of user intentions:

.. code-block:: python

   class QueryIntent(Enum):
       SHOW_FAULTS = "show_faults"
       FILTER_FAULTS = "filter_faults"
       EXPLAIN_FAULT = "explain_fault"
       COUNT_FAULTS = "count_faults"
       ANALYZE_SIGNAL = "analyze_signal"
       HELP = "help"
       UNKNOWN = "unknown"

QueryResult Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Contains the complete result of query processing:

.. code-block:: python

   @dataclass
   class QueryResult:
       intent: QueryIntent           # Identified user intention
       entities: Dict[str, List[str]] # Extracted technical entities
       confidence: float             # Interpretation confidence (0-1)
       response: str                # Human-readable response
       action_data: Optional[Dict]   # Structured action parameters

FaultKnowledgeBase
~~~~~~~~~~~~~~~~~~

Comprehensive knowledge base containing:

* **Fault Definitions**: Technical explanations for each fault type
* **Characteristics**: Behavioral patterns and symptoms
* **Common Causes**: Typical scenarios leading to each fault type

Example fault definition structure:

.. code-block:: python

   'lg': {
       'full_name': 'Line-to-Ground',
       'description': 'A fault between one phase conductor and ground',
       'characteristics': 'Single phase voltage drops to zero or near zero',
       'common_causes': 'Insulation failure, tree contact, equipment failure'
   }

NaturalLanguageProcessor
~~~~~~~~~~~~~~~~~~~~~~~~

The core NLP engine that:

1. **Pattern Matching**: Uses regex patterns for intent recognition
2. **Entity Extraction**: Identifies technical terms and parameters
3. **Confidence Calculation**: Assesses interpretation reliability
4. **Response Generation**: Creates appropriate responses and actions

Intent Recognition Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses regex patterns to identify user intentions:

Show/Display Patterns
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   [
       r'\b(show|display|list|view|get)\b.*\bfaults?\b',
       r'\bfaults?\b.*\b(show|display|list|view)\b',
       r'\b(plot|graph|chart)\b.*\bfaults?\b',
       r'\bgive me.*\bfaults?\b'
   ]

Explanation Patterns
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   [
       r'\b(what|explain|define|describe)\b.*\b(is|are)\b.*\bfault\b',
       r'\b(tell me about|describe)\b.*\bfault\b',
       r'\bfault\b.*\b(definition|explanation|meaning)\b'
   ]

Entity Extraction
~~~~~~~~~~~~~~~~~

The system extracts relevant technical entities:

Fault Type Entities
^^^^^^^^^^^^^^^^^^^

Recognizes various fault type expressions and normalizes them to standard abbreviations.

Phase Entities
^^^^^^^^^^^^^^

Identifies phase references (A, B, C) in various formats:
* "phase A", "A phase"
* "VA", "IA" (voltage/current notation)

Signal Entities
^^^^^^^^^^^^^^^

Recognizes signal type references:
* Voltage, voltages, V
* Current, currents, I
* General signal references

Confidence Score Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system calculates confidence based on:

.. code-block:: python

   def _calculate_confidence(self, intent: QueryIntent, entities: Dict, query: str) -> float:
       base_confidence = 0.7 if intent != QueryIntent.UNKNOWN else 0.2
       entity_boost = min(0.2, len(entities) * 0.1)
       keyword_boost = sum(0.02 for keyword in power_keywords if keyword in query.lower())
       return min(1.0, base_confidence + entity_boost + keyword_boost)

**Factors**:
* **Base Confidence**: 0.7 for recognized intents, 0.2 for unknown
* **Entity Boost**: +0.1 per extracted entity (max +0.2)
* **Keyword Boost**: +0.02 per power system keyword

API Reference
-------------

QueryInterface Class
~~~~~~~~~~~~~~~~~~~~

Main interface for processing natural language queries.

.. code-block:: python

   class QueryInterface:
       def __init__(self)
       def process_query(self, query: str, session_state) -> QueryResult
       def get_suggestions(self, partial_query: str) -> List[str]

Methods
^^^^^^^

process_query(query, session_state)
'''''''''''''''''''''''''''''''''''

**Parameters**:
* ``query`` (str): Natural language query string
* ``session_state``: Streamlit session state object

**Returns**:
* ``QueryResult``: Complete processing result

**Description**: 
Main method for processing natural language queries. Handles intent recognition, entity extraction, confidence calculation, and action execution.

get_suggestions(partial_query)
'''''''''''''''''''''''''''''''

**Parameters**:
* ``partial_query`` (str): Incomplete query string

**Returns**:
* ``List[str]``: List of suggested completions

**Description**:
Provides intelligent query suggestions based on partial input.

NaturalLanguageProcessor Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core NLP processing engine.

.. code-block:: python

   class NaturalLanguageProcessor:
       def __init__(self)
       def process_query(self, query: str) -> QueryResult

FaultKnowledgeBase Class
~~~~~~~~~~~~~~~~~~~~~~~~

Knowledge repository for fault information.

.. code-block:: python

   class FaultKnowledgeBase:
       FAULT_DEFINITIONS = {...}  # Comprehensive fault information

Utility Functions
~~~~~~~~~~~~~~~~~

apply_nlp_filters(results_df, filters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parameters**:
* ``results_df`` (pd.DataFrame): Fault analysis results
* ``filters`` (Dict): Filter criteria from NLP processing

**Returns**:
* ``pd.DataFrame``: Filtered results

**Description**:
Applies NLP-extracted filters to fault analysis data.

create_nlp_response_display(result)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parameters**:
* ``result`` (QueryResult): Processing result to display

**Description**:
Creates appropriate Streamlit display components based on query result type.

Integration Guide
-----------------

Streamlit Integration
~~~~~~~~~~~~~~~~~~~~

The interface integrates seamlessly with Streamlit applications:

.. code-block:: python

   # Initialize in session state
   if 'query_interface' not in st.session_state:
       st.session_state.query_interface = QueryInterface()
   
   # Process user input
   if query_input:
       result = st.session_state.query_interface.process_query(
           query_input, st.session_state
       )
       create_nlp_response_display(result)

Session State Variables
^^^^^^^^^^^^^^^^^^^^^^^

The system uses several session state variables:

* ``query_interface``: Main interface instance
* ``nlp_filters``: Current filter criteria
* ``nlp_action``: Last requested action
* ``nlp_signal_request``: Signal analysis requests
* ``query_input_value``: Current input field value

Data Integration
~~~~~~~~~~~~~~~~

The system expects fault analysis results in the following format:

.. code-block:: python

   {
       'classification_results': [
           {
               'interval': int,
               'start_time': float,
               'end_time': float,
               'duration': float,
               'fault_type': str,  # 'lg', 'llg', 'll', 'lll', 'lllg'
               'confidence': float
           },
           ...
       ]
   }

Customization
-------------

Adding New Intent Types
~~~~~~~~~~~~~~~~~~~~~~~

1. **Add to QueryIntent enum**:

.. code-block:: python

   class QueryIntent(Enum):
       # ... existing intents
       NEW_INTENT = "new_intent"

2. **Add recognition patterns**:

.. code-block:: python

   def _build_intent_patterns(self):
       return {
           # ... existing patterns
           QueryIntent.NEW_INTENT: [
               r'pattern1',
               r'pattern2'
           ]
       }

3. **Add response handler**:

.. code-block:: python

   def _generate_response(self, intent, entities, query):
       # ... existing handlers
       elif intent == QueryIntent.NEW_INTENT:
           return self._handle_new_intent(entities)

Extending Entity Types
~~~~~~~~~~~~~~~~~~~~~~

Add new entity patterns to recognize additional technical terms:

.. code-block:: python

   def _build_entity_patterns(self):
       return {
           # ... existing patterns
           'new_entity_type': [
               r'pattern1',
               r'pattern2'
           ]
       }

Knowledge Base Expansion
~~~~~~~~~~~~~~~~~~~~~~~~

Add new fault types or expand existing definitions:

.. code-block:: python

   FAULT_DEFINITIONS = {
       # ... existing definitions
       'new_fault_type': {
           'full_name': 'New Fault Type',
           'description': 'Technical description',
           'characteristics': 'Behavioral characteristics',
           'common_causes': 'Typical causes'
       }
   }

Best Practices
--------------

Query Writing Tips
~~~~~~~~~~~~~~~~~~

**Be Specific**:
* ✅ "Show LG faults"
* ❌ "Show faults" (too vague)

**Use Standard Terms**:
* ✅ "three-phase faults"
* ✅ "LLL faults" 
* ❌ "big faults"

**Natural Language**:
* ✅ "How many LG faults are there?"
* ✅ "Count LG faults"
* ❌ "LG COUNT"

**Phase References**:
* ✅ "faults with phase A"
* ✅ "phase B faults"
* ❌ "first phase faults"

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Query Complexity**: Simple queries process faster than complex multi-entity queries.

**Pattern Matching**: The system uses regex matching, so very long queries may have slightly higher processing time.

**Memory Usage**: Query history is maintained in session state; consider clearing for long sessions.

Error Handling
~~~~~~~~~~~~~~

The system provides graceful error handling:

**Low Confidence Queries**: 
* Confidence below 0.6 triggers warnings
* Suggestions provided for improvement

**Unrecognized Queries**:
* Clear error messages
* Helpful suggestions
* Link to help documentation

**Data Availability**:
* Checks for analysis completion
* Clear messages when data unavailable

Troubleshooting
--------------- graceful error handling:

**Low Confidence Queries**: 
* Confidence below 0.6 triggers warnings
* Suggestions provided for improvement

**Unrecognized Queries**:
* Clear error messages
* Helpful suggestions
* Link to help documentation

**Data Availability**:
* Checks for analysis completion
* Clear messages when data unavailable



Common Issues
-------------

**"Low confidence" warnings**:
* **Cause**: Ambiguous or non-standard terminology
* **Solution**: Use more specific terms, refer to supported fault types

**"No analysis results available"**:
* **Cause**: Attempting data queries before running analysis
* **Solution**: Complete fault classification analysis first

**Query not understood**:
* **Cause**: Using unsupported terminology or syntax
* **Solution**: Check example queries, use 'help' command

**Empty results**:
* **Cause**: Filters too restrictive or no matching data
* **Solution**: Verify fault types exist in data, broaden criteria

