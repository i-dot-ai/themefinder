"""Cross-cutting themes analysis agent focused on consolidation and group quality.

This module provides the CrossCuttingThemesConsolidationAgent class for performing 
intelligent cross-cutting theme analysis that prioritizes creating meaningful, 
well-consolidated groups over maximizing coverage.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

from langchain.schema.runnable import RunnableWithFallbacks
from tenacity import (
    before,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from themefinder.models import (
    CrossCuttingThemesResponse, 
    CrossCuttingThemeReviewResponse,
)
from themefinder.llm_batch_processor import load_prompt_from_file
from themefinder.themefinder_logging import logger


class ConsolidationStrategy(Enum):
    """Available consolidation strategies for the agent."""
    QUALITY_FOCUSED = "quality_focused"  # Prioritize larger, coherent groups
    BALANCED = "balanced"                 # Balance between quality and coverage
    COMPREHENSIVE = "comprehensive"       # Try to include more themes but maintain quality
    

class CrossCuttingThemesAgent:
    """Agent for performing quality-focused cross-cutting theme analysis.

    This agent prioritizes creating meaningful, well-consolidated theme groups
    over maximizing coverage. It focuses on:
    - Merging related small groups into larger coherent ones
    - Moving themes between groups to strengthen connections
    - Ensuring groups are substantive and meaningful
    """

    def __init__(
        self,
        llm: RunnableWithFallbacks,
        system_prompt: str,
        min_themes: int = 3,
        target_group_size: int = 5,  # Target average themes per group
        max_groups: int = 15,  # Maximum number of groups to maintain quality
        strategy: ConsolidationStrategy = ConsolidationStrategy.BALANCED,
    ):
        """Initialize the consolidation-focused analysis agent.

        Args:
            llm: Language model instance for structured output
            system_prompt: System prompt to guide LLM behavior
            min_themes: Minimum themes required for valid groups
            target_group_size: Target average number of themes per group
            max_groups: Maximum number of groups to prevent fragmentation
            strategy: Consolidation strategy to use
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.min_themes = min_themes
        self.target_group_size = target_group_size
        self.max_groups = max_groups
        self.strategy = strategy
        
        # Agent state
        self.themes_data: Dict[int, Dict] = {}
        self.formatted_themes: List[Dict] = []
        self.total_themes: int = 0
        self.iteration_count: int = 0
        
        # Analysis results
        self.cross_cutting_groups: List[Dict] = []
        self.used_themes: Set[Tuple[int, str]] = set()
        self.unused_themes: List[Dict] = []
        
        # Quality metrics
        self.group_sizes: List[int] = []
        self.average_group_size: float = 0
        self.consolidation_actions: List[str] = []

    def analyze(self, questions_themes: Dict[int, Dict]) -> List[Dict]:
        """Perform quality-focused cross-cutting theme analysis.

        Args:
            questions_themes: Dictionary mapping question numbers to theme data

        Returns:
            List of consolidated cross-cutting theme dictionaries
        """
        self._validate_input(questions_themes)
        self._prepare_data(questions_themes)
        
        logger.info(f"ConsolidationAgent starting with {self.strategy.value} strategy")
        logger.info(f"Target: {self.max_groups} groups, avg size {self.target_group_size}")
        
        # Step 1: Initial analysis (same as original)
        self._step1_initial_analysis()
        
        # Step 2: Consolidation phase - merge small groups
        self._consolidate_small_groups()
        
        # Step 3: Review and redistribute themes  
        self._review_and_redistribute()
        
        # Step 4: Discover additional cross-cutting patterns in unused themes
        self._discover_additional_patterns()
        
        # Step 5: Post-discovery consolidation - merge semantically similar groups
        self._post_discovery_consolidation()
        
        # Step 6: Final consolidation if needed
        if len(self.cross_cutting_groups) > self.max_groups:
            self._final_consolidation()
        
        # Step 7: Validate and fix cross-cutting theme rule violations
        self._validate_and_fix_rule_violations()
        
        # Calculate final metrics
        self._calculate_quality_metrics()
        
        logger.info(f"Final: {len(self.cross_cutting_groups)} groups, "
                   f"avg size {self.average_group_size:.1f}, "
                   f"coverage {self._get_current_coverage():.1%}")
        
        return self.cross_cutting_groups

    def _validate_input(self, questions_themes: Dict[int, Dict]) -> None:
        """Validate input data structure."""
        if not questions_themes:
            raise ValueError("questions_themes cannot be empty")
        
        for question_num, question_data in questions_themes.items():
            if "themes" not in question_data:
                raise KeyError(f"Question {question_num} missing 'themes' key")
            
            themes_df = question_data["themes"]
            required_columns = ["topic_id", "topic_label", "topic_description"]
            missing_columns = [col for col in required_columns if col not in themes_df.columns]
            if missing_columns:
                raise KeyError(f"Question {question_num} themes missing columns: {missing_columns}")

    def _prepare_data(self, questions_themes: Dict[int, Dict]) -> None:
        """Prepare and format theme data for analysis."""
        self.themes_data = questions_themes
        self.formatted_themes = []
        
        for question_num, question_data in questions_themes.items():
            themes_df = question_data["themes"]
            for _, row in themes_df.iterrows():
                self.formatted_themes.append({
                    "question_number": question_num,
                    "theme_key": row["topic_id"],
                    "theme_label": row["topic_label"],
                    "theme_description": row["topic_description"],
                })
        
        self.total_themes = len(self.formatted_themes)
        self.unused_themes = self.formatted_themes.copy()

    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(3),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    def _step1_initial_analysis(self) -> None:
        """Perform initial cross-cutting theme analysis."""
        prompt_template = load_prompt_from_file("cross_cutting_themes")
        themes_data_str = "\n".join([
            f"Question {theme['question_number']}, Theme {theme['theme_key']}: "
            f"{theme['theme_label']} - {theme['theme_description']}"
            for theme in self.formatted_themes
        ])
        
        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            themes_data=themes_data_str
        )
        
        structured_llm = self.llm.with_structured_output(CrossCuttingThemesResponse)
        result = structured_llm.invoke(prompt)
        
        if isinstance(result, dict):
            result = CrossCuttingThemesResponse(**result)
        
        # Store initial groups
        for cc_theme in result.cross_cutting_themes:
            theme_dict = {
                "name": cc_theme.name,
                "description": cc_theme.description,
                "themes": [
                    {
                        "question_number": t.question_number,
                        "theme_key": t.theme_key,
                    }
                    for t in cc_theme.themes
                ],
            }
            self.cross_cutting_groups.append(theme_dict)
            
            for theme in theme_dict["themes"]:
                self.used_themes.add((theme["question_number"], theme["theme_key"]))
        
        self._update_unused_themes()
        logger.info(f"Initial analysis: {len(self.cross_cutting_groups)} groups found")

    def _consolidate_small_groups(self) -> None:
        """Consolidate small groups by merging related ones."""
        if not self.cross_cutting_groups:
            return
            
        # Identify small groups (below target size)
        small_groups = [g for g in self.cross_cutting_groups if len(g["themes"]) < self.target_group_size]
        
        if not small_groups or len(small_groups) < 2:
            return
        
        logger.info(f"Consolidating {len(small_groups)} small groups")
        
        # Use LLM to identify which small groups can be merged
        consolidation_prompt = self._create_consolidation_prompt(small_groups)
        
        # For now, we'll use a heuristic approach
        # In a full implementation, we'd use the LLM to suggest mergers
        self._heuristic_consolidation(small_groups)

    def _heuristic_consolidation(self, small_groups: List[Dict]) -> None:
        """Apply heuristic rules to consolidate small groups."""
        # Sort small groups by size (largest first)
        small_groups.sort(key=lambda x: len(x["themes"]), reverse=True)
        
        consolidated = []
        used_indices = set()
        
        for i, group1 in enumerate(small_groups):
            if i in used_indices:
                continue
                
            # Try to find a compatible group to merge with
            for j, group2 in enumerate(small_groups[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # Check if merging would create a good-sized group
                combined_size = len(group1["themes"]) + len(group2["themes"])
                if combined_size <= self.target_group_size * 1.5:  # Don't create oversized groups
                    # Check for name/description similarity (simple heuristic)
                    if self._groups_are_related(group1, group2):
                        # Merge the groups
                        merged = self._merge_groups(group1, group2)
                        consolidated.append(merged)
                        used_indices.add(i)
                        used_indices.add(j)
                        self.consolidation_actions.append(
                            f"Merged '{group1['name']}' with '{group2['name']}'"
                        )
                        break
        
        # Remove merged groups from main list and add consolidated ones
        self.cross_cutting_groups = [
            g for i, g in enumerate(self.cross_cutting_groups) 
            if g not in small_groups or small_groups.index(g) not in used_indices
        ]
        self.cross_cutting_groups.extend(consolidated)

    def _groups_are_related(self, group1: Dict, group2: Dict) -> bool:
        """Check if two groups are semantically related using improved heuristics."""
        # Define strong semantic indicators for common themes
        strong_indicators = {
            "public_ownership": {"public", "ownership", "nationalization", "nationalisation", 
                               "renationalisation", "renationalization", "state", "control"},
            "regulation": {"regulation", "regulatory", "enforcement", "compliance", "oversight", 
                          "monitoring", "penalties", "powers"},
            "investment": {"investment", "infrastructure", "funding", "finance", "capital"},
            "transparency": {"transparency", "accountability", "reporting", "disclosure"},
            "environment": {"environmental", "climate", "sustainability", "biodiversity", "health"},
            "consumer": {"consumer", "customer", "affordability", "pricing", "protection", "cost"},
            "consultation": {"consultation", "involvement", "engagement", "participation", "public"}
        }
        
        # Extract key concepts from both groups
        group1_text = f"{group1['name']} {group1['description']}".lower()
        group2_text = f"{group2['name']} {group2['description']}".lower()
        
        # Check for strong semantic matches
        for theme_type, indicators in strong_indicators.items():
            group1_matches = sum(1 for word in indicators if word in group1_text)
            group2_matches = sum(1 for word in indicators if word in group2_text)
            
            # If both groups have strong indicators for the same theme type, they're related
            # But be more selective - need higher match count to avoid over-consolidation
            if group1_matches >= 3 and group2_matches >= 3:
                return True
        
        # Fallback to original word overlap logic but with better threshold
        name1_words = set(group1["name"].lower().split())
        name2_words = set(group2["name"].lower().split())
        desc1_words = set(group1["description"].lower().split())
        desc2_words = set(group2["description"].lower().split())
        
        # Remove common words
        common_words = {"and", "the", "of", "in", "to", "for", "a", "an", "on", "with", 
                       "about", "that", "this", "these", "those", "are", "is", "be", "or"}
        
        name1_words -= common_words
        name2_words -= common_words
        desc1_words -= common_words
        desc2_words -= common_words
        
        name_overlap = name1_words & name2_words
        desc_overlap = desc1_words & desc2_words
        
        # Conservative overlap threshold - need strong name similarity
        return len(name_overlap) > 1 or (len(name_overlap) > 0 and len(desc_overlap) > 2)

    def _merge_groups(self, group1: Dict, group2: Dict) -> Dict:
        """Merge two groups into one consolidated group."""
        # Combine themes
        merged_themes = group1["themes"] + group2["themes"]
        
        # Smart name merging based on semantic similarity
        name1_lower = group1["name"].lower()
        name2_lower = group2["name"].lower()
        
        # Check for common concepts to create better merged names
        if "public ownership" in name1_lower and "public ownership" in name2_lower:
            merged_name = "Public Ownership and Nationalization"
        elif "regulation" in name1_lower and "regulation" in name2_lower:
            merged_name = "Regulatory Effectiveness and Enforcement"
        elif "accountability" in name1_lower and "transparency" in name2_lower:
            merged_name = "Accountability and Transparency"
        elif "investment" in name1_lower and "infrastructure" in name2_lower:
            merged_name = "Investment and Infrastructure"
        elif "consumer" in name1_lower and ("affordability" in name2_lower or "pricing" in name2_lower):
            merged_name = "Consumer Protection and Affordability"
        elif "environment" in name1_lower and ("health" in name2_lower or "sustainability" in name2_lower):
            merged_name = "Environmental and Public Health"
        elif len(merged_name := f"{group1['name']} and {group2['name']}") <= 60:
            # Use combined name if not too long
            pass
        else:
            # Use the more comprehensive name (larger group)
            merged_name = group1["name"] if len(group1["themes"]) >= len(group2["themes"]) else group2["name"]
        
        # Merge descriptions intelligently
        desc1 = group1["description"]
        desc2 = group2["description"]
        
        # Find common themes in descriptions to avoid redundancy
        if any(word in desc1.lower() and word in desc2.lower() 
               for word in ["accountability", "transparency", "public", "ownership", "regulation"]):
            # Descriptions overlap significantly - create a unified description
            merged_desc = f"{desc1} This encompasses related themes including {desc2.lower()}"
        else:
            # Descriptions are complementary
            merged_desc = f"{desc1} Additionally, this includes {desc2.lower()}"
        
        return {
            "name": merged_name,
            "description": merged_desc,
            "themes": merged_themes
        }

    def _review_and_redistribute(self) -> None:
        """Review unused themes and redistribute to strengthen existing groups."""
        if not self.unused_themes or not self.cross_cutting_groups:
            return
            
        prompt_template = load_prompt_from_file("cross_cutting_themes_review")
        
        existing_themes_str = "\n".join([
            f"- {cc['name']}: {cc['description']}\n  Current themes: " + 
            ", ".join([f"Q{t['question_number']}-{t['theme_key']}" for t in cc['themes']])
            for cc in self.cross_cutting_groups
        ])
        
        unused_themes_str = "\n".join([
            f"Q{t['question_number']}-{t['theme_key']}: {t['theme_label']} - {t['theme_description']}"
            for t in self.unused_themes
        ])
        
        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            existing_themes=existing_themes_str,
            unused_themes=unused_themes_str
        )
        
        try:
            structured_llm = self.llm.with_structured_output(CrossCuttingThemeReviewResponse)
            result = structured_llm.invoke(prompt)
            
            if isinstance(result, dict):
                result = CrossCuttingThemeReviewResponse(**result)
        except Exception as e:
            logger.warning(f"Review and redistribute failed: {e}")
            # Skip this step if it fails to avoid breaking the whole process
            return
        
        # Apply additions to existing groups
        for addition in result.additions:
            target_group = self._find_group_by_name(addition.cross_cutting_theme_name)
            if target_group and self._can_add_theme(target_group, addition):
                target_group["themes"].append({
                    "question_number": addition.question_number,
                    "theme_key": addition.theme_key
                })
                self.used_themes.add((addition.question_number, addition.theme_key))
                self.consolidation_actions.append(
                    f"Added Q{addition.question_number}-{addition.theme_key} to '{target_group['name']}'"
                )
        
        self._update_unused_themes()

    def _discover_additional_patterns(self) -> None:
        """Discover new cross-cutting patterns in unused themes."""
        if not self.unused_themes or len(self.unused_themes) < 6:
            logger.info("Insufficient unused themes for pattern discovery")
            return
        
        logger.info(f"Discovering additional patterns in {len(self.unused_themes)} unused themes")
        
        # Create prompt for pattern discovery
        prompt_template = self._create_pattern_discovery_prompt()
        
        unused_themes_str = "\n".join([
            f"Q{t['question_number']}-{t['theme_key']}: {t['theme_label']} - {t['theme_description']}"
            for t in self.unused_themes
        ])
        
        existing_groups_str = "\n".join([
            f"- {group['name']}: {group['description']} ({len(group['themes'])} themes)"
            for group in self.cross_cutting_groups
        ])
        
        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            unused_themes=unused_themes_str,
            existing_groups=existing_groups_str,
            min_themes=self.min_themes,
            target_group_size=self.target_group_size
        )
        
        try:
            structured_llm = self.llm.with_structured_output(CrossCuttingThemesResponse)
            result = structured_llm.invoke(prompt)
            
            if isinstance(result, dict):
                result = CrossCuttingThemesResponse(**result)
            
            # Add discovered patterns as new groups
            new_groups_added = 0
            for new_pattern in result.cross_cutting_themes:
                if len(new_pattern.themes) >= self.min_themes:
                    # Check that themes span multiple questions
                    unique_questions = set(t.question_number for t in new_pattern.themes)
                    if len(unique_questions) >= 3:  # Must span at least 3 questions
                        new_group = {
                            "name": new_pattern.name,
                            "description": new_pattern.description,
                            "themes": [
                                {
                                    "question_number": t.question_number,
                                    "theme_key": t.theme_key,
                                }
                                for t in new_pattern.themes
                            ],
                        }
                        
                        # Verify all themes are actually unused
                        valid_themes = []
                        for theme in new_group["themes"]:
                            theme_tuple = (theme["question_number"], theme["theme_key"])
                            if theme_tuple not in self.used_themes:
                                valid_themes.append(theme)
                                self.used_themes.add(theme_tuple)
                        
                        if len(valid_themes) >= self.min_themes:
                            new_group["themes"] = valid_themes
                            self.cross_cutting_groups.append(new_group)
                            new_groups_added += 1
                            self.consolidation_actions.append(
                                f"Discovered new pattern: '{new_pattern.name}' ({len(valid_themes)} themes)"
                            )
            
            if new_groups_added > 0:
                logger.info(f"Discovered {new_groups_added} new cross-cutting patterns")
                self._update_unused_themes()
            else:
                logger.info("No new cross-cutting patterns discovered")
                
        except Exception as e:
            logger.warning(f"Pattern discovery failed: {e}")

    def _post_discovery_consolidation(self) -> None:
        """Consolidate semantically similar groups after pattern discovery."""
        if len(self.cross_cutting_groups) < 2:
            return
        
        logger.info(f"Post-discovery consolidation: checking {len(self.cross_cutting_groups)} groups for semantic similarity")
        
        # Find pairs of related groups
        groups_to_merge = []
        already_merged = set()
        
        for i, group1 in enumerate(self.cross_cutting_groups):
            if i in already_merged:
                continue
                
            for j, group2 in enumerate(self.cross_cutting_groups[i+1:], i+1):
                if j in already_merged:
                    continue
                    
                if self._groups_are_related(group1, group2):
                    # Check if merging would violate size constraints
                    combined_size = len(group1["themes"]) + len(group2["themes"])
                    if combined_size <= self.target_group_size * 2:  # Don't create overly large groups
                        groups_to_merge.append((i, j, group1, group2))
                        already_merged.add(i)
                        already_merged.add(j)
                        break
        
        # Perform merges
        if groups_to_merge:
            # Sort by indices in reverse order to maintain indexing
            groups_to_merge.sort(key=lambda x: x[1], reverse=True)
            
            for i, j, group1, group2 in groups_to_merge:
                # Merge the groups
                merged = self._merge_groups(group1, group2)
                
                # Remove original groups (remove higher index first)
                self.cross_cutting_groups.pop(j)
                self.cross_cutting_groups.pop(i)
                
                # Add merged group
                self.cross_cutting_groups.append(merged)
                
                self.consolidation_actions.append(
                    f"Post-discovery merge: '{group1['name']}' + '{group2['name']}' → '{merged['name']}'"
                )
                
                logger.info(f"Merged '{group1['name']}' with '{group2['name']}' → '{merged['name']}'")

    def _create_pattern_discovery_prompt(self) -> str:
        """Create a prompt for discovering new cross-cutting patterns."""
        return '''
{system_prompt}

You have already identified several cross-cutting themes from survey data. Now analyze the remaining unused themes to discover any additional cross-cutting patterns that were missed.

EXISTING CROSS-CUTTING GROUPS (already identified):
{existing_groups}

UNUSED THEMES TO ANALYZE:
{unused_themes}

TASK: Review the unused themes above and identify any NEW cross-cutting theme groups that:
1. Span themes from at least 3 different questions
2. Have at least {min_themes} themes total
3. Represent a coherent policy area or concern
4. Are genuinely distinct from the existing groups above

CRITERIA FOR SUCCESS:
- Look for themes that share similar policy concerns across multiple questions
- Target around {target_group_size} themes per group for optimal size
- Only create groups that represent genuine cross-cutting patterns
- Avoid forcing unrelated themes together just to create groups
- Focus on meaningful policy groupings that would be useful for analysis

If no new cross-cutting patterns exist in the unused themes, return an empty list.

Provide your analysis as cross-cutting theme groups with names, descriptions, and constituent themes.
'''

    def _final_consolidation(self) -> None:
        """Perform final consolidation if there are still too many groups."""
        while len(self.cross_cutting_groups) > self.max_groups:
            # Find the two smallest groups
            self.cross_cutting_groups.sort(key=lambda x: len(x["themes"]))
            
            if len(self.cross_cutting_groups) < 2:
                break
                
            smallest = self.cross_cutting_groups[0]
            second_smallest = self.cross_cutting_groups[1]
            
            # Merge them
            merged = self._merge_groups(smallest, second_smallest)
            
            # Remove the original groups and add the merged one
            self.cross_cutting_groups = self.cross_cutting_groups[2:]
            self.cross_cutting_groups.append(merged)
            
            self.consolidation_actions.append(
                f"Final consolidation: merged '{smallest['name']}' with '{second_smallest['name']}'"
            )

    def _validate_and_fix_rule_violations(self) -> None:
        """Validate cross-cutting theme rule and fix violations using LLM."""
        logger.info("Validating cross-cutting theme rule compliance")
        
        violations_found = 0
        groups_fixed = 0
        
        for group in self.cross_cutting_groups[:]:  # Create copy to allow modification
            # Check for rule violations (multiple themes from same question)
            question_themes = defaultdict(list)
            for theme in group["themes"]:
                question_themes[theme["question_number"]].append(theme)
            
            # Find violations
            violated_questions = {
                q: themes for q, themes in question_themes.items() 
                if len(themes) > 1
            }
            
            if violated_questions:
                violations_found += sum(len(themes) - 1 for themes in violated_questions.values())
                groups_fixed += 1
                
                logger.info(f"Fixing rule violations in '{group['name']}': "
                           f"{len(violated_questions)} questions with multiple themes")
                
                # Use LLM to select best theme from each violated question
                fixed_group = self._fix_group_violations(group, violated_questions)
                
                # Update the group in place
                group.update(fixed_group)
        
        if violations_found > 0:
            logger.info(f"Fixed {violations_found} rule violations across {groups_fixed} groups")
            # Recalculate used themes after fixes
            self.used_themes = set()
            for group in self.cross_cutting_groups:
                for theme in group["themes"]:
                    self.used_themes.add((theme["question_number"], theme["theme_key"]))
            self._update_unused_themes()
        else:
            logger.info("No rule violations found - all groups comply with cross-cutting theme rule")

    def _fix_group_violations(self, group: Dict, violated_questions: Dict) -> Dict:
        """Use LLM to select best theme from each violated question."""
        try:
            # Build prompt with violation details
            prompt = self._create_violation_fix_prompt(group, violated_questions)
            
            # Get LLM decision on which themes to keep
            structured_llm = self.llm.with_structured_output(CrossCuttingThemeReviewResponse)
            result = structured_llm.invoke(prompt)
            
            if isinstance(result, dict):
                result = CrossCuttingThemeReviewResponse(**result)
            
            # Build new theme list based on LLM selections
            new_themes = []
            
            # Keep themes from non-violated questions
            for theme in group["themes"]:
                if theme["question_number"] not in violated_questions:
                    new_themes.append(theme)
            
            # Add LLM-selected themes from violated questions
            for addition in result.additions:
                if addition.question_number in violated_questions:
                    # Check if this theme was one of the original options
                    original_theme_keys = {t["theme_key"] for t in violated_questions[addition.question_number]}
                    if addition.theme_key in original_theme_keys:
                        new_themes.append({
                            "question_number": addition.question_number,
                            "theme_key": addition.theme_key
                        })
                        self.consolidation_actions.append(
                            f"Rule fix: Selected Q{addition.question_number}-{addition.theme_key} "
                            f"for '{group['name']}' (was {len(violated_questions[addition.question_number])} themes)"
                        )
            
            return {
                "name": group["name"],
                "description": group["description"],
                "themes": new_themes
            }
            
        except Exception as e:
            logger.warning(f"LLM-based violation fix failed for '{group['name']}': {e}")
            # Fallback: keep first theme from each violated question
            return self._fallback_fix_violations(group, violated_questions)

    def _create_violation_fix_prompt(self, group: Dict, violated_questions: Dict) -> str:
        """Create prompt for LLM to fix rule violations."""
        violation_details = []
        
        for q_num, themes in violated_questions.items():
            theme_details = []
            for theme in themes:
                details = self._get_theme_details_for_prompt(theme["question_number"], theme["theme_key"])
                theme_details.append(f"    • {theme['theme_key']}: {details['label']}")
                theme_details.append(f"      {details['description']}")
            
            violation_details.append(f"  Question {q_num} ({len(themes)} themes - VIOLATION):")
            violation_details.extend(theme_details)
        
        return f"""
{self.system_prompt}

CROSS-CUTTING THEME RULE VIOLATION FIX

You are reviewing a cross-cutting theme group that violates the fundamental rule:
"Each cross-cutting group should contain at most ONE theme per question"

GROUP TO FIX:
Name: {group['name']}
Description: {group['description']}

RULE VIOLATIONS FOUND:
{chr(10).join(violation_details)}

TASK: For each question that has multiple themes, select the ONE theme that:
1. Best represents the cross-cutting concept described in the group
2. Has the strongest thematic connection to themes from other questions
3. Most clearly demonstrates the cross-cutting nature of this policy area

IMPORTANT: You must select exactly ONE theme per violated question.
Return your selections using the standard addition format, specifying which theme from each violated question should be KEPT in the group.

The goal is to maintain the thematic coherence while ensuring each question contributes only one theme to this cross-cutting pattern.
"""

    def _get_theme_details_for_prompt(self, question_number: int, theme_key: str) -> Dict[str, str]:
        """Get theme details for prompt generation."""
        for theme in self.formatted_themes:
            if (theme["question_number"] == question_number and 
                theme["theme_key"] == theme_key):
                return {
                    'label': theme['theme_label'],
                    'description': theme['theme_description']
                }
        return {'label': 'Unknown', 'description': 'Theme not found'}

    def _fallback_fix_violations(self, group: Dict, violated_questions: Dict) -> Dict:
        """Fallback method to fix violations by keeping first theme per question."""
        new_themes = []
        
        # Keep themes from non-violated questions
        for theme in group["themes"]:
            if theme["question_number"] not in violated_questions:
                new_themes.append(theme)
        
        # Keep first theme from each violated question
        for q_num, themes in violated_questions.items():
            selected_theme = themes[0]  # Just take first one
            new_themes.append(selected_theme)
            self.consolidation_actions.append(
                f"Rule fix (fallback): Kept Q{q_num}-{selected_theme['theme_key']} "
                f"for '{group['name']}' (removed {len(themes)-1} themes)"
            )
        
        return {
            "name": group["name"],
            "description": group["description"],  
            "themes": new_themes
        }

    def _calculate_quality_metrics(self) -> None:
        """Calculate quality metrics for the final groups."""
        # Remove groups below minimum threshold
        self.cross_cutting_groups = [
            group for group in self.cross_cutting_groups
            if len(group["themes"]) >= self.min_themes
        ]
        
        # Sort by size (largest first)
        self.cross_cutting_groups.sort(key=lambda x: len(x["themes"]), reverse=True)
        
        # Calculate metrics
        self.group_sizes = [len(g["themes"]) for g in self.cross_cutting_groups]
        self.average_group_size = sum(self.group_sizes) / len(self.group_sizes) if self.group_sizes else 0

    def _get_current_coverage(self) -> float:
        """Calculate current theme coverage percentage."""
        return len(self.used_themes) / self.total_themes if self.total_themes > 0 else 0.0

    def _update_unused_themes(self) -> None:
        """Update the list of unused themes based on current used themes."""
        self.unused_themes = [
            theme for theme in self.formatted_themes
            if (theme["question_number"], theme["theme_key"]) not in self.used_themes
        ]

    def _find_group_by_name(self, name: str) -> Optional[Dict]:
        """Find cross-cutting group by name."""
        for group in self.cross_cutting_groups:
            if group["name"] == name:
                return group
        return None

    def _can_add_theme(self, group: Dict, addition: Any) -> bool:
        """Check if a theme can be added to a group (no duplicate questions)."""
        existing_questions = {t["question_number"] for t in group["themes"]}
        return addition.question_number not in existing_questions

    def _create_consolidation_prompt(self, small_groups: List[Dict]) -> str:
        """Create a prompt for LLM-guided consolidation."""
        groups_str = "\n".join([
            f"{i+1}. {g['name']} ({len(g['themes'])} themes): {g['description']}"
            for i, g in enumerate(small_groups)
        ])
        
        return f"""
        Review these small cross-cutting theme groups and suggest which ones could be 
        meaningfully consolidated into larger, more coherent groups:
        
        {groups_str}
        
        Goal: Create fewer but more substantive groups with {self.target_group_size} or more themes each.
        Focus on semantic relationships and policy coherence rather than just combining unrelated groups.
        """

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get detailed summary of the consolidation process."""
        return {
            "strategy_used": self.strategy.value,
            "final_groups": len(self.cross_cutting_groups),
            "average_group_size": self.average_group_size,
            "group_sizes": self.group_sizes,
            "final_coverage": self._get_current_coverage(),
            "total_themes": self.total_themes,
            "used_themes": len(self.used_themes),
            "unused_themes": len(self.unused_themes),
            "consolidation_actions": self.consolidation_actions,
            "quality_score": self._calculate_quality_score(),
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate a quality score based on group characteristics."""
        if not self.cross_cutting_groups:
            return 0.0
            
        # Factors for quality:
        # 1. Average group size (closer to target is better)
        size_score = 1.0 - abs(self.average_group_size - self.target_group_size) / self.target_group_size
        size_score = max(0, min(1, size_score))
        
        # 2. Number of groups (closer to target is better)
        ideal_groups = min(self.max_groups, self.total_themes // self.target_group_size)
        group_count_score = 1.0 - abs(len(self.cross_cutting_groups) - ideal_groups) / ideal_groups
        group_count_score = max(0, min(1, group_count_score))
        
        # 3. Distribution uniformity (less variance is better)
        if len(self.group_sizes) > 1:
            variance = sum((s - self.average_group_size) ** 2 for s in self.group_sizes) / len(self.group_sizes)
            std_dev = variance ** 0.5
            uniformity_score = 1.0 - (std_dev / self.average_group_size) if self.average_group_size > 0 else 0
            uniformity_score = max(0, min(1, uniformity_score))
        else:
            uniformity_score = 1.0
        
        # Weighted average
        quality_score = (size_score * 0.4 + group_count_score * 0.3 + uniformity_score * 0.3)
        
        return round(quality_score, 3)