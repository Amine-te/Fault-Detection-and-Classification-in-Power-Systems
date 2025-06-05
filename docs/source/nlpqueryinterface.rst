NLP Query Interface
======================================

Overview
--------

A natural language processing interface for power system fault analysis using spaCy. Enables users to query fault data using plain English commands.

Features
--------

* **Intent Recognition**: Identifies query types (show, explain, count, filter)
* **Entity Extraction**: Recognizes fault types and phases
* **Knowledge Base**: Built-in fault definitions and explanations
* **Smart Filtering**: Apply filters based on natural language queries
* **Query History**: Track and review previous queries

Supported Query Types
---------------------

Show/Filter Faults
~~~~~~~~~~~~~~~~~~

* "Show all LG faults"
* "Display three-phase faults"
* "List faults with phase A"

Explain Faults
~~~~~~~~~~~~~~

* "What is an LG fault?"
* "Explain line-to-ground faults"
* "Tell me about LLG faults"

Count Faults
~~~~~~~~~~~~

* "How many LG faults?"
* "Count three-phase faults"

Fault Types
-----------

+------+-----------------------+--------------------------------+
| Code | Full Name             | Description                    |
+======+=======================+================================+
| LG   | Line-to-Ground        | Single phase to ground fault   |
+------+-----------------------+--------------------------------+
| LLG  | Line-to-Line-to-Ground| Two phases and ground fault    |
+------+-----------------------+--------------------------------+
| LL   | Line-to-Line          | Two phase fault                |
+------+-----------------------+--------------------------------+
| LLL  | Three-Phase           | Balanced three-phase fault     |
+------+-----------------------+--------------------------------+
| LLLG | Three-Phase-to-Ground | All phases and ground fault    |
+------+-----------------------+--------------------------------+

Core Classes
------------

QueryIntent
~~~~~~~~~~~

Enumeration of supported query types:

* ``SHOW_FAULTS``: Display fault data
* ``EXPLAIN_FAULT``: Get fault explanations
* ``COUNT_FAULTS``: Count fault occurrences
* ``FILTER_FAULTS``: Apply data filters

SpacyNLPProcessor
~~~~~~~~~~~~~~~~~

Main NLP processing engine:

.. code-block:: python

   processor = SpacyNLPProcessor()
   result = processor.process_query("Show all LG faults")

QueryInterface
~~~~~~~~~~~~~~

User interface for query processing:

.. code-block:: python

   interface = QueryInterface()
   result = interface.process_query(query, session_state)

Installation Requirements
-------------------------

.. code-block:: bash

   pip install spacy streamlit pandas
   python -m spacy download en_core_web_sm

Usage Example
-------------

.. code-block:: python

   from spacy_nlp_processor import QueryInterface
   
   # Initialize interface
   interface = QueryInterface()
   
   # Process natural language query
   result = interface.process_query("Show all LG faults", session_state)
   
   # Display results
   print(f"Intent: {result.intent}")
   print(f"Response: {result.response}")
   print(f"Confidence: {result.confidence}")

API Reference
-------------

QueryResult
~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class QueryResult:
       intent: QueryIntent
       entities: Dict[str, List[str]]
       confidence: float
       response: str
       action_data: Optional[Dict] = None

Key Methods
~~~~~~~~~~~

``process_query(query: str) -> QueryResult``
    Process natural language query and return structured result.

``get_suggestions(partial_query: str) -> List[str]``
    Generate intelligent query suggestions based on partial input.

``apply_nlp_filters(df: DataFrame, filters: Dict) -> DataFrame``
    Apply extracted filters to fault data.

Configuration
-------------

Pattern Matching
~~~~~~~~~~~~~~~~

The system uses spaCy's pattern matching for:

* Intent recognition (show, explain, count)
* Entity extraction (fault types, phases)
* Semantic analysis for confidence scoring

Confidence Scoring
~~~~~~~~~~~~~~~~~~

Factors affecting confidence:

* Power system terminology usage
* Entity extraction success
* Query grammatical structure
* Query length and specificity

Limitations
-----------

* Requires spaCy model ``en_core_web_sm``
* Limited to predefined fault types
* English language only
* Requires pre-analyzed data for filtering queries

Integration
-----------

Designed for Streamlit applications with session state management. Integrates with existing fault analysis workflows and provides natural language interface overlay.