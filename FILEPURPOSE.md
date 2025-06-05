1. adaptive_alphawall.py

AdaptiveAlphaWall (class)

__init__
_load_thresholds
_save_thresholds
_load_feedback
_save_feedback
_load_calibration
_detect_emotional_state
_calculate_context_score
_detect_intent
record_feedback
_adapt_thresholds
learn_pattern
add_false_positive
get_adaptation_stats


upgrade_to_adaptive_alphawall
create_feedback_handler

2. adaptive_migration.py

evaluate_link_with_confidence_gates
AdaptiveThresholds (class)

__init__
_load_age
_save_age
bump
get_migration_threshold


MigrationEngine (class)

__init__
should_migrate
find_similar_items
should_migrate_by_overlap
migrate_from_bridge
get_migration_summary



3. adaptive_quarantine_layer.py

AdaptiveQuarantine (class)

__init__
_load_adaptive_config
_save_adaptive_config
_load_feedback
_save_feedback
_calculate_vagueness_score
_detect_true_recursion
_extract_text_pattern
_are_varied_questions
should_quarantine_with_learning
_update_decision_context
record_feedback
_save_to_log
_learn_from_false_positive
get_adaptive_stats
reset_session_context


adaptive_quarantine_check

4. alphawall.py

AlphaWall (class)

__init__
_init_vault
_generate_memory_id
_store_in_vault
_detect_emotional_state
_detect_intent
_detect_context_type
_assess_risk_flags
_generate_embedding_similarity
process_input
_suggest_routing
_calculate_routing_confidence
_save_zone_output
get_zone_output_by_id
clear_recursion_window
get_vault_stats


create_alphawall_handler

5. alphawall_bridge_adapter.py

AlphaWallBridgeAdapter (class)

__init__
_load_tag_mappings
process_user_input
_tags_to_parser_config
_generate_synthetic_input
_parse_with_alphawall_context
_calculate_logic_score_from_tags
_calculate_symbolic_score_from_tags
_evaluate_with_tags
_determine_response_strategy
_record_decision
get_routing_stats
_get_top_n


create_alphawall_pipeline
integrate_with_existing_parser

6. autonomous_learner.py

load_adaptive_weights
store_to_tripartite_memory
evaluate_link_with_confidence_gates
initialize_data_files_if_needed
load_deferred_urls
save_deferred_urls
save_curriculum_metrics
score_text_against_keywords
score_text_for_symbolic_relevance
log_link_decision
evaluate_link_action
process_chunk_to_tripartite
autonomous_learning_cycle

7. brain_metrics.py

BrainMetrics (class)

__init__
_write_json
_read_json
log_decision
save_session_metrics
generate_report
get_adaptive_weights
analyze_conflicts


display_metrics_summary

8. bridge_adapter.py

AlphaWallBridge (class)

__init__
_load_tag_weights
_calculate_tag_adjustments
process_with_alphawall
_save_decision
learn_from_feedback
get_decision_pattern_analysis


create_alphawall_aware_bridge

9. cluster_namer.py

generate_cluster_id
pick_cluster_name
generate_cluster_name
load_symbols
extract_texts
cluster_symbols
summarize_cluster
assign_cluster_names

10. clustering.py

cluster_memory

11. content_utils.py

detect_content_type

12. decision_history.py

HistoryAwareMemory (class)

__init__
store
_get_current_weights
get_item_stability
get_items_by_stability


NoOpLock (class)

acquire
release
__enter__
__exit__



13. download_models.py
(No functions - just model downloading script)
14. emotion_handler.py

predict_emotions

15. graph_visualizer.py
(No functions - just visualization script)
16. inspect_vectors.py

inspect_memory

17. linguistic_warfare.py

LinguisticWarfareDetector (class)

__init__
_load_attack_patterns
analyze_text_for_warfare
_detect_pattern_threats
_analyze_structure
_analyze_semantics
_analyze_temporal_patterns
_check_recursive_depth
_detect_contradictions
_get_user_risk_profile
_calculate_threat_score
_determine_defense_strategy
_generate_defense_explanation
_log_defense_action
_update_user_profile
_save_attack_patterns
_save_defense_log
_save_user_profiles
_load_defense_log
_load_user_profiles
get_defense_statistics
learn_from_attack


check_for_warfare

18. link_evaluator.py

EnhancedLinkEvaluator (class)

__init__
evaluate_with_full_pipeline
_determine_severity
_apply_contamination_adjustments
_make_final_decision
_save_decision
provide_feedback
_update_weight_evolution
get_system_health


evaluate_link_with_confidence_gates
evaluate_with_alphawall

19. link_utils.py

evaluate_link_with_confidence_gates

20. main.py

is_url
generate_response
main

21. memory_analytics.py

MemoryAnalyzer (class)

__init__
get_memory_stats
_calculate_avg_score
_calculate_avg_age
_calculate_stability
_calculate_volatility
_calculate_health_indicators
analyze_bridge_patterns
generate_evolution_report
_save_report
print_report



22. memory_architecture.py

TripartiteMemory (class)

__init__
_load_all
_load_safe
store
save_all
_save_safe
get_counts
clear_all



23. memory_evolution_engine.py

MemoryEvolutionEngine (class)

__init__
run_evolution_cycle
_print_distribution
_get_performance_stats
_log_session
add_test_data


run_memory_evolution

24. memory_maintenance.py

score_text_against_phase1_keywords
prune_phase1_symbolic_vectors

25. memory_optimizer.py

load_adaptive_config
save_adaptive_config
recompute_adaptive_link_weights
is_url
extract_new_emojis
perform_acceptance_resolution
process_web_url_placeholder
generate_response
main

26. parser.py

load_seed_symbols
load_emotion_map
extract_keywords
chunk_content
extract_symbolic_units
parse_input
parse_with_emotion
is_zone_output
parse_raw_text

These modules create a comprehensive AI system with:

Security layers (AlphaWall, Quarantine, Warfare Detection)
Memory management (Tripartite architecture, Evolution, Migration)
Content processing (Parser, Emotion handling, Symbol management)
Analytics and monitoring (Brain metrics, Memory analytics, Visualization)
Learning capabilities (Adaptive weights, Pattern recognition, Feedback loops)

1. parser.py

load_seed_symbols(file_path)
load_emotion_map(file_path)
extract_keywords(text_input, max_keywords=10)
chunk_content(text, max_chunk_size=1000, overlap=100)
extract_symbolic_units(input_data, current_lexicon)
parse_input(input_data, current_lexicon=None)
parse_with_emotion(input_data, detected_emotions_verified, current_lexicon)
is_zone_output(input_data)
parse_raw_text(text_input, current_lexicon=None)

2. processing_nodes.py

_load_cooccurrence_log()
_save_cooccurrence_log(cooccurrence_data)
_update_symbol_cooccurrence(symbolic_node_output)
evaluate_link_with_confidence_gates(logic_score, symbolic_score, logic_scale=10.0, sym_scale=5.0)
detect_content_type(text_input, spacy_nlp_instance=None)
initialize_processing_nodes()

Classes and their methods:
LogicNode:

__init__(self, vector_memory_path_str=None)
store_memory(self, text_input, source_url=None, source_type="web_scrape", ...)
retrieve_memories(self, query_text, current_phase_directives)

SymbolicNode:

__init__(self, seed_symbols_path_str, symbol_memory_path_str, ...)
_ensure_data_files(self)
_load_meta_symbols(self)
_save_meta_symbols(self)
_get_active_symbol_lexicon(self, current_phase_directives)
evaluate_chunk_symbolically(self, chunk_text, current_phase_directives_for_lexicon)
process_input_for_symbols(self, text_input, detected_emotions_output, ...)
run_meta_symbol_analysis(self, max_phase_to_consider)

CurriculumManager:

__init__(self)
get_current_phase(self)
get_max_phases(self)
get_phase_context_description(self, phase)
get_processing_directives(self, phase)
update_metrics(self, phase, chunks_processed_increment=0, ...)
advance_phase_if_ready(self, current_completed_phase_num)
get_all_metrics(self)

DynamicBridge:

__init__(self, logic_node, symbolic_node, curriculum_manager)
_load_adaptive_weights(self)
_detect_emotions(self, text_input)
_score_text_for_phase(text_content, phase_directives) (static method)
is_chunk_relevant_for_current_phase(self, text_chunk, ...)
determine_target_storage_phase(self, text_chunk, current_processing_phase_num)
route_chunk_for_processing(self, text_input, source_url, ...)
generate_response_for_user(self, user_input_text, source_url=None)
get_routing_statistics(self)

3. quarantine_layer.py

should_quarantine_input(source_type, source_url=None)

UserMemoryQuarantine class:

__init__(self, data_dir="data")
_init_files(self)
get_quarantine_statistics(self)
quarantine(self, zone_id, reason="manual_quarantine", severity="medium")
check_user_history(self, user_id)
load_all_quarantined_memory(self)
check_contamination_risk(self, zone_output)
quarantine_user_input(self, text, user_id, source_url=None, ...)

4. reverse_migration.py
ReverseMigrationAuditor class:

__init__(self, memory, confidence_threshold=0.3)
audit_item(self, item, current_location)
audit_logic_memory(self)
audit_symbolic_memory(self)
audit_all(self)
get_audit_summary(self)

5. run_pipeline.py

run_learning_pipeline(data_dir="data", learning_config=None, evolution_config=None, cycles=1)
main()

6. symbol_chainer.py

load_memory()
build_symbol_chains(min_similarity=0.4)
print_symbol_chains()

7. symbol_cluster.py

cluster_vectors_and_plot(show_graph=True)
dummy_generate_cluster_id(texts)

8. symbol_drift_plot.py

show_symbol_drift(symbol_filter=None)

9. symbol_emotion_cluster.py

show_emotion_clusters()

10. symbol_emotion_updater.py

load_emotion_map(file_path=DEFAULT_MAP_PATH)
save_emotion_map(emotion_map, file_path=DEFAULT_MAP_PATH)
update_symbol_emotions(matched_symbols_weighted, verified_emotions, file_path=DEFAULT_MAP_PATH)

11. symbol_generator.py

generate_symbol_from_context(text, keywords, emotions_list_of_tuples)
mock_extract_keywords(text_input) (in test section)

12. symbol_memory.py

load_symbol_memory(file_path=SYMBOL_MEMORY_PATH)
save_symbol_memory(memory, file_path=SYMBOL_MEMORY_PATH)
_check_quarantine_status(origin, example_text=None, name=None, keywords=None)
_sanitize_symbol_data(symbol_data)
add_symbol(symbol_token, name, keywords, initial_emotions, example_text, ...)
get_symbol_details(symbol_token, file_path=SYMBOL_MEMORY_PATH)
update_symbol_emotional_profile(symbol_token, emotion_changes, file_path=SYMBOL_MEMORY_PATH)
prune_duplicates(file_path=SYMBOL_MEMORY_PATH)
get_emotion_profile(symbol_token, file_path=SYMBOL_MEMORY_PATH)
get_golden_memory(symbol_token, file_path=SYMBOL_MEMORY_PATH)
_get_symbol_color(emotions)
_calculate_display_priority(resonance_weight, origin)
_get_classification_hint(keywords, emotions)
validate_symbol_token(symbol_token)
get_symbols_for_visualization(limit=100, min_usage=0, exclude_quarantined=True, file_path=SYMBOL_MEMORY_PATH)
quarantine_existing_symbol(symbol_token, reason="manual_quarantine", file_path=SYMBOL_MEMORY_PATH)

13. symbol_suggester.py

load_vectors()
save_symbol(symbol_obj)
suggest_symbols_from_vectors(min_cluster_size=3)

14. symbol_test.ipynb (Jupyter notebook functions)

describe_symbol(token)
search_by_keyword(keyword)
save_symbol_to_memory(token, context="", emotion="")
show_memory()
search_memory(query)
symbol_frequency_report()
emotion_cluster_report()
trace_symbol(token)
detect_emergent_loops(min_emotions=2, min_occurrences=3)
bind_meta_symbol(original_token, new_token, name, summary)

15. system_analytics.py

plot_node_activation_timeline(trail_log_path=TRAIL_LOG_PATH_SA)
plot_symbol_popularity_timeline(occurrence_log_path=OCCURRENCE_LOG_PATH_SA, top_n_symbols=7)
plot_curriculum_metrics(metrics_path=CURRICULUM_METRICS_PATH_SA)

Other Files (JSON data files):

symbol_memory.json - Contains symbol definitions and metadata
symbolic_memory.json.backup - Backup of symbolic memory data

Note: Some files reference imported modules that aren't fully shown in the documents (like emotion_handler, trail_log, user_memory, vector_engine, vector_memory, etc.), so their internal functions aren't listed here.

1. talk_to_ai.py

generate_response(user_input: str, processing_result: dict) -> str
display_system_state(result: dict, processing_time: float)
process_user_input(user_input: str) -> dict
get_memory_stats_safe(node, node_type="unknown")
get_tripartite_summary_safe()
show_stats()
show_memory_distribution()
show_current_weights()
run_test_scenarios()
main()
patched_detect_emotional_state(text) (patched function)

2. trail_graph.py

show_trail_graph()

3. trail_log.py

_load_log()
_save_log(log_entries)
log_dynamic_bridge_processing_step(log_id=None, text_input=None, source_url=None, current_phase=0, directives=None, is_highly_relevant_for_phase=False, target_storage_phase_for_chunk=None, is_shallow_content=False, content_type_heuristic="ambiguous", detected_emotions_output=None, logic_node_output=None, symbolic_node_output=None, generated_response_preview=None)
log_trail(text, symbols, matches, file_path=TRAIL_LOG_FILE_PATH)
add_emotions(entry_id, emotions, file_path=TRAIL_LOG_FILE_PATH)

4. tripartite_dashboard.py
Classes:

EventLogger:

__init__(self, log_dir="logs", buffer_size=100)
log_event(self, event_type: str, data: Dict[str, Any]) -> None
_flush_buffer(self) -> None
log_bridge_decision(self, url: str, logic_score: float, symbol_score: float, decision: str) -> None
log_symbol_activation(self, symbol: str, emotion: str, resonance: float, context: str, origin_trace: Optional[Dict] = None) -> None
log_memory_migration(self, item_id: str, from_node: str, to_node: str, reason: str, bridge_event_id: Optional[str] = None) -> None



Functions:

validate_schema(data: Any, expected_type: type, schema_name: str = "") -> bool
safe_load_json(filename: str, expected_type: type = dict) -> Any
compute_memory_metrics(symbol_memory: dict, logic_memory: list, bridge_memory: list) -> dict
convert_to_jsonl(json_file: str, output_dir: str = "logs") -> Optional[Path]

5. upgrade_old_vectors.py

upgrade_vectors()

6. user_memory.py

load_user_memory(file_path=DEFAULT_USER_MEMORY_PATH)
save_user_memory(entries, file_path=DEFAULT_USER_MEMORY_PATH)
add_user_memory_entry(symbol, context_text, emotion_in_context, source_url=None, learning_phase=None, is_context_highly_relevant=None, file_path=DEFAULT_USER_MEMORY_PATH)

7. vector_engine.py

encode_with_minilm(text: str) -> np.ndarray
encode_with_e5(text: str) -> np.ndarray
fuse_vectors(text: str, threshold: float = 0.7)
embed_text(text: str, model_choice: str = "minilm")

8. vector_memory.py

_load_memory() -> List[Dict[str, Any]]
_save_memory(memory_data: List[Dict[str, Any]])
load_vectors(path: Optional[Path] = None) -> List[Dict[str, Any]]
save_vectors(vectors: List[Dict[str, Any]], path: Optional[Path] = None)
store_vector(text: str, source_url: Optional[str] = None, source_type: str = "unknown", learning_phase: int = 0, exploration_depth: str = "shallow", confidence: float = 0.5, source_trust: str = "unknown", metadata: Optional[Dict] = None) -> Dict[str, Any]
_store_quarantined_vector(text: str, entry_id: str, source_url: Optional[str], source_type: str, learning_phase: int, quarantine_result: Dict, warfare_analysis: Optional[Dict] = None) -> bool
retrieve_similar_vectors(query_text: str, top_n: int = 5, max_phase_allowed: int = 999, min_confidence: float = 0.0, include_quarantined: bool = False, similarity_threshold: float = 0.0) -> List[Tuple[float, Dict]]
get_memory_stats() -> Dict[str, Any]
prepare_memory_for_visualization(entry_id: str) -> Optional[Dict]
cleanup_quarantined_vectors(days_old: int = 30) -> Dict[str, int]
get_quarantine_summary() -> Dict[str, Any]
store(*args, **kwargs) (backward compatibility wrapper)
retrieve(*args, **kwargs) (backward compatibility wrapper)

9. visualization_prep.py
Class: VisualizationPrep

__init__(self, data_dir="data", enable_nlp: bool = True)
_update_quarantine_cache(self)
_check_quarantine_trace_risk(self, segment_text: str) -> Tuple[bool, Optional[str]]
prepare_text_for_display(self, text: str, processing_result: Dict, include_emotions: bool = True, include_symbols: bool = True, include_entities: bool = True) -> Dict
_segment_text(self, text: str) -> List[str]
_create_base_segment(self, index: int, text: str) -> Dict
_analyze_segment_enhanced(self, segment_data: Dict, segment_text: str, global_result: Dict, zone_analysis: Dict, bridge_decision: Dict, contamination_check: Dict, include_emotions: bool, include_symbols: bool, include_entities: bool) -> Dict
_detect_content_type(self, text: str) -> str
_calculate_logic_score(self, text: str) -> float
_calculate_symbolic_score(self, text: str) -> float
_add_bridge_info(self, segment: Dict, bridge_decision: Dict) -> Dict
_add_zone_tags(self, segment: Dict, zone_analysis: Dict) -> Dict
_add_recursion_info(self, segment: Dict, zone_analysis: Dict) -> Dict
_add_contamination_info(self, segment: Dict, contamination: Dict) -> Dict
_add_emotion_analysis(self, segment: Dict, text: str) -> Dict
_add_entity_extraction(self, segment: Dict, text: str) -> Dict
_add_symbol_analysis(self, segment: Dict, text: str) -> Dict
_calculate_segment_confidence(self, segment: Dict) -> float
_generate_viz_hints(self, segment: Dict) -> Dict
_build_hover_data_enhanced(self, segment_text: str, segment_data: Dict, include_emotions: bool, include_symbols: bool) -> Dict
_get_classification_reason(self, segment_data: Dict) -> str
_get_confidence_explanation(self, segment_data: Dict) -> str
_get_symbol_tooltip(self, symbol: str) -> str
_generate_summary(self, segments: List[Dict], processing_result: Dict) -> Dict
_empty_result(self) -> Dict
_save_visualization(self, viz_output: Dict)
generate_html_preview(self, viz_output: Dict) -> str
_build_hover_html(self, segment: Dict) -> str
_generate_enhanced_css(self) -> str
generate_json_for_react(self, viz_output: Dict) -> str
create_emotion_overlay(self, segments: List[Dict]) -> List[Dict]
create_risk_overlay(self, segments: List[Dict]) -> List[Dict]
_get_emotion_color(self, emotion: str) -> str
export_for_frontend(self, viz_data: Dict, format: str = 'json') -> str

Standalone functions:

visualize_processing_result(text: str, processing_result: Dict) -> Dict
detect_content_type(text: str, nlp=None) -> str

10. web_parser.py

fetch_raw_html(url, timeout=DEFAULT_TIMEOUT)
extract_links_with_text_from_html(base_url, html_content)
clean_html_to_text(html_content, use_trafilatura_on_string=False)
fetch_shallow(url, max_chars=500, timeout=DEFAULT_TIMEOUT-5)
fetch_and_clean_with_trafilatura(url, timeout_not_used=None)
fallback_extract_text_with_bs4(url, timeout=DEFAULT_TIMEOUT)
chunk_text(text, max_chunk_length=1000, overlap=100)

11. weight_evolution.py
Class: WeightEvolver

__init__(self, data_dir="data")
_load_weights(self)
_save_weights(self)
_load_momentum(self)
_save_momentum(self)
_load_history(self)
_save_history(self)
get_current_specialization(self)
calculate_target_specialization(self, run_count, memory_stats=None)
evolve_weights(self, run_count, memory_stats=None, performance_stats=None)
get_evolution_summary(self)

This represents a comprehensive AI system with components for:

Memory management (vector storage, symbol memory, user memory)
Text processing (parsing, chunking, visualization preparation)
Decision routing (bridge decisions, weight evolution)
Monitoring (dashboard, logging, trail tracking)
Web interaction (parsing, link extraction)
AI interaction (main talk_to_ai interface)
