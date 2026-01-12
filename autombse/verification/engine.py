from __future__ import annotations

from datetime import datetime
import json
import os
import re


class BDDRules:
    def __init__(self, package_tree):
        self.package_tree = package_tree
        self.errors = []
        self.warnings = []
        self.erules = [
            {"rule_id": "BDD-001", "description": "All Blocks must be named and unique", "check_function": self.validate_unique_naming},
            {"rule_id": "BDD-002", "description": "Properties must have types", "check_function": self.validate_property_type},
            {"rule_id": "BDD-003", "description": "Generalization can only be used between Blocks", "check_function": self.validate_generalization_usage},
            {"rule_id": "BDD-004", "description": "Composition/Aggregation can only connect Parts and Blocks", "check_function": self.validate_composition_aggregation},
            {"rule_id": "BDD-008", "description": "A Block's Parts must have concrete types (not abstract)", "check_function": self.validate_part_type},
            {"rule_id": "BDD-009", "description": "A Block port must not connect directly to another Block's attribute", "check_function": self.validate_port_connection},
            {"rule_id": "BDD-010", "description": "All internal behaviors must define clear inputs/outputs", "check_function": self.validate_behavior_definition},
            {"rule_id": "BDD-012", "description": "Interface Blocks should not contain implemented behaviors", "check_function": self.validate_interface_block_behavior},
            {"rule_id": "BDD-013", "description": "Block attributes must specify data types and be compatible with external systems", "check_function": self.validate_attribute_data_type},
            {"rule_id": "BDD-015", "description": "Block generalization should form a valid inheritance chain", "check_function": self.validate_generalization_chain}
        ]
        self.wrules = [
            {"rule_id": "BDD-005", "description": "Blocks should define clear interfaces (Port/FlowPort/ProxyPort)", "check_function": self.validate_port},
        ]

    def validate(self):
        for rule in self.erules:
            valid, errs  = rule["check_function"](self.package_tree)
            if not valid:
                self.errors.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "errors": errs
                })

        for rule in self.wrules:
            valid, warnings = rule["check_function"](self.package_tree)
            if not valid:
                self.warnings.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "warnings": warnings
                })

        return self.errors, self.warnings

    def validate_unique_naming(self, package_tree):
        """
        Validates BDD-001: All Blocks must be named and unique.
        """
        names = []
        errors = []

        for package in package_tree:
            if package.type == "block":
                if package.name in names:
                    errors.append(f"Duplicate Block name: {package.name}")
                elif package.name == None:
                    errors.append(f"No Block name")
                names.append(package.name)
            if package.children:
                valid, error = self.validate_unique_naming(package.children)
                errors += error

        return len(errors) ==0, errors

    def validate_property_type(self, package_tree):
        """
        Validates BDD-002: Properties must have types.
        """
        errors = []

        for package in package_tree:
            if package.type == "block":
                for child in package.children:
                    if child.type == "part" and child.belongsto == "None":
                        errors.append(f"Property {child.name} of block {package.name} has no type defined")
            if package.children:
                valid, error = self.validate_property_type(package.children)
                errors += error

        return len(errors) ==0, errors

    def validate_generalization_usage(self, package_tree):
        """
        Validates BDD-003: Generalization can only be used between Blocks.
        """
        errors = []

        for package in package_tree:
            if hasattr(package, "generalization") and package.generalization and package.type != "block":
                errors.append(f"Generalization used on non-Block element: {package.name}")

        return len(errors) ==0, errors

    def validate_composition_aggregation(self, package_tree):
        """
        Validates BDD-004: Composition/Aggregation can only connect Parts and Blocks.
        """
        errors = []

        for package in package_tree:
            if hasattr(package, "composition") and package.composition:
                if package.type != "part" and package.type != "block":
                    errors.append(f"Composition Aggregation used on non-Part/Block element: {package.name}")

        return len(errors) ==0, errors
    
    def validate_port(self, package_tree):
        """
        Validates BDD-005: Blocks should define clear interfaces.
        """

        warnings = []
        pkgs = {}

        for index, package in enumerate(package_tree):
            if package.type == "block":
                flag = False
                for child in package.children:
                    if child.name == None:
                        continue
                    elif child.type == "port":
                        flag = True
                        pkgs[package.name] = True
                    elif child.type == "part":
                        if child.belongsto in pkgs.keys():
                            if pkgs[child.belongsto]:
                                flag = True
                                pkgs[package.name] = True
                        elif "port" in str.lower(child.name):
                                flag = True
                                pkgs[package.name] = True
        
                if not flag:
                    warnings.append(f"Block {package.name} has no ports")
            
            if package.children:
                valid, warning = self.validate_port(package.children)
                warnings += warning
         
        return len(warnings) == 0, warnings


    def validate_part_type(self, package_tree):
        """
        Validates BDD-008: A Block's Parts must have concrete types (not abstract).
        """
        errors = []
        pkgs = []

        for package in package_tree:
            pkgs.append(package)
            if package.type == "part":
                part_type = package.belongsto
                if part_type != None:
                    for pkg in pkgs:
                        if pkg.name == part_type:
                            t = pkg.type
                            if t != "block" and t != "part":
                                errors.append(f"Part {package.name} belongs to {part_type} with {t} has no type specified")
                            break

        return len(errors) ==0, errors

    def validate_port_connection(self, package_tree):
        """
        Validates BDD-009: A Block port must not connect directly to another Block's attribute.
        """
        errors = []

        for package in package_tree:
            if package.type == "port" and hasattr(package, "connections"):
                for connection in package.connections:
                    if connection.target.type == "attribute":
                        errors.append(f"Port {package.name} directly connected to attribute {connection.target.name}")

        return len(errors) ==0, errors

    def validate_behavior_definition(self, package_tree):
        """
        Validates BDD-010: All internal behaviors must define clear inputs/outputs.
        """
        errors = []

        for package in package_tree:
            if package.type == "block" and hasattr(package, "behaviors"):
                for behavior in package.behaviors:
                    if not hasattr(behavior, "inputs") or not hasattr(behavior, "outputs"):
                        errors.append(f"Behavior {behavior.name} in Block {package.name} has no input/output definition")

        return len(errors) ==0, errors

    def validate_interface_block_behavior(self, package_tree):
        """
        Validates BDD-012: Interface Blocks should not contain implemented behaviors.
        """
        errors = []

        for package in package_tree:
            if package.type == "interface" and hasattr(package, "behaviors"):
                for behavior in package.behaviors:
                    if behavior.implementation:
                        errors.append(f"Interface Block {package.name} contains implemented behavior {behavior.name}")

        return len(errors) ==0, errors

    def validate_attribute_data_type(self, package_tree):
        """
        Validates BDD-013: Block attributes must specify data types and be compatible with external systems.
        """
        errors = []

        for package in package_tree:
            if package.type == "block":
                for child in package.children:
                    if child.belongsto == "None":
                        errors.append(f" {child.name} in Block {package.name} has no data type specified")

        return len(errors) ==0, errors

    def validate_generalization_chain(self, package_tree):
        """
        Validates BDD-015: Block generalization should form a valid inheritance chain.
        """
        errors = []

        for package in package_tree:
            if package.type == "block" and hasattr(package, "generalizations"):
                for generalization in package.generalizations:
                    if generalization.circular:
                        errors.append(f"Block {package.name} has circular generalization")
                    if not generalization.valid_hierarchy:
                        errors.append(f"Block {package.name} has invalid generalization hierarchy")

        return len(errors) ==0, errors
    
class CrossRules:
    def __init__(self, package_tree):
        self.package_tree = package_tree
        self.errors = []
        self.warnings = []
        self.rules = [
            # RD rules
            {"rule_id": "RD-1", "description": "Requirement constraints must map to design-block attributes or behaviors", 
             "check_function": self.validate_requirement_constraints},
            {"rule_id": "RD-2", "description": "Requirements must have corresponding implementations in BDD/IBD", 
             "check_function": self.validate_requirement_implementation},
            
            # BDD rules
            {"rule_id": "BBD-1", "description": "Attributes defined in BDD must be instantiated in IBD", 
             "check_function": self.validate_bdd_attributes_instantiation},
            
            # IBD rules
            {"rule_id": "IBD-1", "description": "All part connections must be defined in the BDD", 
             "check_function": self.validate_ibd_connections},
            {"rule_id": "IBD-2", "description": "Signal types used in IBD connections must be defined in the BDD", 
             "check_function": self.validate_ibd_signal_types},
            {"rule_id": "IBD-3", "description": "Ports defined in IBD must have corresponding implementation blocks in the BDD", 
             "check_function": self.validate_ibd_ports},
            
            # AD rules
            {"rule_id": "AD-1", "description": "Actions and parts in the activity diagram must be implemented in the BDD", 
             "check_function": self.validate_ad_implementations},
            {"rule_id": "AD-2", "description": "Guard conditions must be reflected in requirements", 
             "check_function": self.validate_ad_guards},
            
            # SMD rules
            {"rule_id": "SMD-1", "description": "If a state machine contains states, corresponding blocks and behaviors must be defined in BDD/IBD", 
             "check_function": self.validate_smd_states},
            {"rule_id": "SMD-2", "description": "State machine transitions must be consistent with actions defined in the activity diagram", 
             "check_function": self.validate_smd_transitions},
            {"rule_id": "SMD-3", "description": "Event-triggered transitions must have corresponding behavior implementations in other views (e.g., BDD/IBD)", 
             "check_function": self.validate_smd_events},
            {"rule_id": "SMD-4", "description": "Each state in the state machine should have a corresponding behavioral description in the activity diagram", 
             "check_function": self.validate_smd_state_behaviors},
            
            # PD rules
            {"rule_id": "PD-1", "description": "Constraints must match attributes in design blocks and be valid", 
             "check_function": self.validate_pd_constraints}
        ]

    def validate(self):
        """Execute all rule checks"""
        # Check if package_tree is empty
        if not self.package_tree:
            self.errors.append({
                "rule_id": "EMPTY-TREE",
                "description": "The model tree to be checked is empty",
                "warnings": ["No model elements found for checking"]
            })
            return self.errors, self.warnings

        # Check if there are any valid elements
        has_valid_elements = False
        for package in self.package_tree:
            if package.type in ["requirement", "block", "part", "interface", "activity", "state_machine", "parametric"]:
                has_valid_elements = True
                break

        if not has_valid_elements:
            self.errors.append({
                "rule_id": "NO-VALID-ELEMENTS",
                "description": "No valid model elements found",
                "warnings": ["The model tree does not contain any requirements, blocks, parts, interfaces, activities, state machines, or parametric diagram elements"]
            })
            return self.errors, self.warnings

        # Execute rule checks
        for rule in self.rules:
            valid, errs = rule["check_function"]()
            if not valid:
                self.errors.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "errors": errs
                })
        return self.errors, self.warnings

    def validate_requirement_constraints(self):
        """Validate that requirement constraints map to design-block attributes or behaviors."""
        errors = []
        warnings = []

        def norm_ref(value) -> str:
            text = ("" if value is None else str(value)).strip()
            if not text:
                return ""
            text = text.rstrip(";").strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
                text = text[1:-1].strip()
            # Prefer unqualified identifiers for matching with the parse tree (which is typically unqualified).
            text = re.split(r"::|\.", text)[-1]
            text = re.sub(r"[^\w]+$", "", text)
            return text

        # Gather satisfied requirements from SysML `satisfy` relations in the combined model.
        satisfied_requirements: set[str] = set()
        for node in self.package_tree:
            if getattr(node, "type", None) != "satisfy":
                continue
            req_name = norm_ref(getattr(node, "name", None))
            if req_name:
                satisfied_requirements.add(req_name)

        def walk(node):
            for child in getattr(node, "children", []) or []:
                yield child
                yield from walk(child)

        # Map constraint nodes to their parent requirement (when nested under a requirement).
        constraint_parent_req: dict[int, str] = {}
        for node in self.package_tree:
            if getattr(node, "type", None) != "requirement":
                continue
            req_name = norm_ref(getattr(node, "name", None))
            if not req_name:
                continue
            for child in walk(node):
                if getattr(child, "type", None) == "constraint":
                    constraint_parent_req[id(child)] = req_name

        seen: set[tuple[str, str]] = set()
        for package in self.package_tree:
            if getattr(package, "type", None) != "constraint":
                continue
            constraint_name = norm_ref(getattr(package, "name", None))
            parent_req = constraint_parent_req.get(id(package), "")
            key = (constraint_name, parent_req)
            if key in seen:
                continue
            seen.add(key)

            # If the parent requirement is satisfied via explicit `satisfy`, do not force a fragile name match.
            if parent_req and parent_req in satisfied_requirements:
                continue
            # If the constraint is nested under an unsatisfied requirement, let RD-2 report the failure.
            if parent_req and parent_req not in satisfied_requirements:
                continue

            # Fallback for standalone constraints: find a corresponding attribute/action by name.
            found = False
            for other_package in self.package_tree:
                if getattr(other_package, "type", None) in ["attribute", "action"]:
                    if self._find_matching_implementation(other_package, getattr(package, "name", None)):
                        found = True
                        break

            if not found:
                errors.append(
                    {
                        "rule_id": "RD-1",
                        "description": f"Requirement constraint '{package.name}' has no corresponding design implementation",
                        "location": f"package {package.name}",
                    }
                )
        
        return len(errors) == 0, errors
    
    def validate_requirement_implementation(self):
        """Validate that requirements have corresponding implementations in BDD/IBD."""
        errors = []
        warnings = []

        def norm_ref(value) -> str:
            text = ("" if value is None else str(value)).strip()
            if not text:
                return ""
            text = text.rstrip(";").strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
                text = text[1:-1].strip()
            text = re.split(r"::|\.", text)[-1]
            text = re.sub(r"[^\w]+$", "", text)
            return text

        satisfied_requirements: set[str] = set()
        for node in self.package_tree:
            if getattr(node, "type", None) != "satisfy":
                continue
            req_name = norm_ref(getattr(node, "name", None))
            if req_name:
                satisfied_requirements.add(req_name)

        seen: set[str] = set()
        for package in self.package_tree:
            if getattr(package, "type", None) != "requirement":
                continue
            req_name = norm_ref(getattr(package, "name", None))
            if not req_name or req_name in seen:
                continue
            seen.add(req_name)

            if req_name in satisfied_requirements:
                continue

            # Legacy fallback: match requirement name to a design element name (works for small toy models).
            found = False
            for other_package in self.package_tree:
                if getattr(other_package, "type", None) in ["part", "action", "block"]:
                    if self._find_matching_implementation(other_package, getattr(package, "name", None)):
                        found = True
                        break

            if not found:
                errors.append(
                    {
                        "rule_id": "RD-2",
                        "description": f"Requirement '{package.name}' has no corresponding implementation",
                        "location": f"package {package.name}",
                    }
                )
        
        return len(errors) == 0, errors
    
    def validate_bdd_attributes_instantiation(self):
        """Validate that attributes defined in the BDD are instantiated in the IBD."""
        errors = []
        warnings = []

        def norm_ref(value) -> str:
            text = ("" if value is None else str(value)).strip()
            if not text:
                return ""
            text = text.rstrip(";").strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
                text = text[1:-1].strip()
            text = re.split(r"::|\.", text)[-1]
            text = re.sub(r"[^\w]+$", "", text)
            return text

        attr_by_owner: list[tuple[str, str]] = []
        for node in self.package_tree:
            if getattr(node, "type", None) not in ["block", "part"]:
                continue
            children = getattr(node, "children", []) or []
            if getattr(node, "type", None) != "block":
                if not any(getattr(child, "type", None) in ["part", "port", "connector"] for child in children):
                    continue

            owner = norm_ref(getattr(node, "belongsto", None)) or norm_ref(getattr(node, "name", None))
            if not owner:
                continue
            for child in children:
                if getattr(child, "type", None) != "attribute":
                    continue
                # Skip attribute definitions (often written as `attribute def X { ... }`) which have children.
                if getattr(child, "children", None):
                    continue
                attr_name = norm_ref(getattr(child, "name", None))
                if attr_name:
                    attr_by_owner.append((owner, attr_name))

        if not attr_by_owner:
            return True, []

        instantiated_types: set[str] = set()
        for node in self.package_tree:
            if getattr(node, "type", None) != "part":
                continue
            belongs_to = norm_ref(getattr(node, "belongsto", None))
            if belongs_to:
                instantiated_types.add(belongs_to)

        seen: set[tuple[str, str]] = set()
        for owner, attr_name in attr_by_owner:
            key = (owner, attr_name)
            if key in seen:
                continue
            seen.add(key)
            if owner in instantiated_types:
                continue
            errors.append(
                {
                    "rule_id": "BBD-1",
                    "description": f"BDD-defined attribute '{attr_name}' is not instantiated",
                    "location": f"package {owner}::{attr_name}",
                }
            )
        
        return len(errors) == 0, errors
    
    def validate_ibd_connections(self):
        """Validate that IBD connectors have explicit names."""
        errors = []
        warnings = []

        for package in self.package_tree:
            if getattr(package, "type", None) != "connector":
                continue
            name = str(getattr(package, "name", "") or "").strip()
            if not name or name.startswith("unnamed_connector_"):
                errors.append(
                    {
                        "rule_id": "IBD-1",
                        "description": "Connector is missing an explicit name (unnamed connector)",
                        "location": f"package {name or '<unnamed>'}",
                    }
                )
                continue
        
        return len(errors) == 0, errors
    
    def _find_matching_implementation(self, package, name):
        """Recursively find a matching implementation."""
        if package.name == name:
            return True
            
        for child in package.children:
            if self._find_matching_implementation(child, name):
                return True
                
        return False
    
    def validate_ibd_signal_types(self):
        """Validate that signal types used in IBD connections are defined in the BDD."""
        errors = []
        warnings = []
        
        for package in self.package_tree:
            if package.type == 'signal':
                # Find signal type definition
                found = False
                for other_package in self.package_tree:
                    if other_package.type in ['part', 'port']:
                        if self._find_matching_implementation(other_package, package.name):
                            found = True
                            break
                
                if not found:
                    errors.append({
                        'rule_id': 'IBD-2',
                        'description': f"Signal type '{package.name}' is not defined in the BDD",
                        'location': f"package {package.name}"
                    })
        
        return len(errors) == 0, errors
    
    def validate_ibd_ports(self):
        """Validate that ports defined in IBD have corresponding implementation blocks in the BDD."""
        errors = []
        warnings = []

        for package in self.package_tree:
            if getattr(package, "type", None) != "port":
                continue
            name = str(getattr(package, "name", "") or "").strip()
            if not name or name.startswith("unnamed_port_"):
                errors.append(
                    {
                        "rule_id": "IBD-3",
                        "description": "Port is missing an explicit name (unnamed port)",
                        "location": f"package {name or '<unnamed>'}",
                    }
                )
                continue
        
        return len(errors) == 0, errors

    def validate_ad_implementations(self):
        """AD-1: Actions and parts in the activity diagram must be implemented in the BDD."""
        errors = []
        for package in self.package_tree:
            if package.type == "activity":
                if not hasattr(package, 'actions'):
                    self.warnings.append({
                        "rule_id": "AD-1",
                        "description": "Activity missing actions",
                        "warnings": [f"Activity {package.name} does not have actions"]
                    })
                    continue
                for action in package.actions:
                    found_implementation = False
                    for block in self.package_tree:
                        if block.type == "block":
                            if not hasattr(block, 'behaviors'):
                                self.warnings.append({
                                    "rule_id": "AD-1",
                                    "description": "Block missing behaviors",
                                    "warnings": [f"Block {block.name} does not have behaviors"]
                                })
                                continue
                            if action.name in block.behaviors:
                                found_implementation = True
                    if not found_implementation:
                        errors.append(f"Action {action.name} in activity {package.name} has no implementation in BDD")
        return len(errors) == 0, errors

    def validate_ad_guards(self):
        """AD-2: Guard conditions must be reflected in requirements."""
        errors = []
        for package in self.package_tree:
            if package.type == "activity":
                if not hasattr(package, 'transitions'):
                    self.warnings.append({
                        "rule_id": "AD-2",
                        "description": "Activity missing transitions",
                        "warnings": [f"Activity {package.name} does not have transitions"]
                    })
                    continue
                for transition in package.transitions:
                    if not hasattr(transition, 'guard'):
                        self.warnings.append({
                            "rule_id": "AD-2",
                            "description": "Transition missing guard",
                            "warnings": [f"Transition in activity {package.name} does not have guard"]
                        })
                        continue
                    if transition.guard:
                        found_requirement = False
                        for req in self.package_tree:
                            if req.type == "requirement":
                                if not hasattr(req, 'constraints'):
                                    self.warnings.append({
                                        "rule_id": "AD-2",
                                        "description": "Requirement missing constraints",
                                        "warnings": [f"Requirement {req.name} does not have constraints"]
                                    })
                                    continue
                                if transition.guard in req.constraints:
                                    found_requirement = True
                        if not found_requirement:
                            errors.append(f"Guard '{transition.guard}' in activity {package.name} is not reflected in requirements")
        return len(errors) == 0, errors

    def validate_smd_states(self):
        """SMD-1: If a state machine contains states, corresponding blocks and behaviors must be defined in BDD/IBD."""
        errors = []
        for package in self.package_tree:
            if package.type == "state_machine":
                if not hasattr(package, 'states'):
                    self.warnings.append({
                        "rule_id": "SMD-1",
                        "description": "State machine missing states",
                        "warnings": [f"State machine {package.name} does not have states"]
                    })
                    continue
                for state in package.states:
                    found_implementation = False
                    for block in self.package_tree:
                        if block.type in ["block", "part"]:
                            if not hasattr(block, 'states'):
                                self.warnings.append({
                                    "rule_id": "SMD-1",
                                    "description": "Block/Part missing states",
                                    "warnings": [f"{block.type} {block.name} does not have states"]
                                })
                                continue
                            if state.name in block.states:
                                found_implementation = True
                    if not found_implementation:
                        errors.append(f"State {state.name} in state machine {package.name} has no corresponding implementation in BDD/IBD")
        return len(errors) == 0, errors

    def validate_smd_transitions(self):
        """SMD-2: State machine transitions must be consistent with actions defined in the activity diagram."""
        errors = []
        for package in self.package_tree:
            if package.type == "state_machine":
                if not hasattr(package, 'transitions'):
                    self.warnings.append({
                        "rule_id": "SMD-2",
                        "description": "State machine missing transitions",
                        "warnings": [f"State machine {package.name} does not have transitions"]
                    })
                    continue
                for transition in package.transitions:
                    if not hasattr(transition, 'action'):
                        self.warnings.append({
                            "rule_id": "SMD-2",
                            "description": "Transition missing action",
                            "warnings": [f"Transition in state machine {package.name} does not have action"]
                        })
                        continue
                    found_action = False
                    for activity in self.package_tree:
                        if activity.type == "activity":
                            if not hasattr(activity, 'actions'):
                                self.warnings.append({
                                    "rule_id": "SMD-2",
                                    "description": "Activity missing actions",
                                    "warnings": [f"Activity {activity.name} does not have actions"]
                                })
                                continue
                            if transition.action in activity.actions:
                                found_action = True
                    if not found_action:
                        errors.append(f"Transition action {transition.action} in state machine {package.name} has no corresponding action in Activity Diagram")
        return len(errors) == 0, errors

    def validate_smd_events(self):
        """SMD-3: Event-triggered transitions must have corresponding implementations in other views (e.g., BDD/IBD)."""
        errors = []
        for package in self.package_tree:
            if package.type == "state_machine":
                if not hasattr(package, 'transitions'):
                    self.warnings.append({
                        "rule_id": "SMD-3",
                        "description": "State machine missing transitions",
                        "warnings": [f"State machine {package.name} does not have transitions"]
                    })
                    continue
                for transition in package.transitions:
                    if not hasattr(transition, 'event'):
                        self.warnings.append({
                            "rule_id": "SMD-3",
                            "description": "Transition missing event",
                            "warnings": [f"Transition in state machine {package.name} does not have event"]
                        })
                        continue
                    if transition.event:
                        found_implementation = False
                        for block in self.package_tree:
                            if block.type in ["block", "part"]:
                                if not hasattr(block, 'events'):
                                    self.warnings.append({
                                        "rule_id": "SMD-3",
                                        "description": "Block/Part missing events",
                                        "warnings": [f"{block.type} {block.name} does not have events"]
                                    })
                                    continue
                                if transition.event in block.events:
                                    found_implementation = True
                        if not found_implementation:
                            errors.append(f"Event {transition.event} in state machine {package.name} has no corresponding implementation in BDD/IBD")
        return len(errors) == 0, errors

    def validate_smd_state_behaviors(self):
        """SMD-4: Each state in the state machine should have a corresponding behavioral description in the activity diagram."""
        errors = []
        for package in self.package_tree:
            if package.type == "state_machine":
                if not hasattr(package, 'states'):
                    self.warnings.append({
                        "rule_id": "SMD-4",
                        "description": "State machine missing states",
                        "warnings": [f"State machine {package.name} does not have states"]
                    })
                    continue
                for state in package.states:
                    found_behavior = False
                    for activity in self.package_tree:
                        if activity.type == "activity":
                            if not hasattr(activity, 'blocks'):
                                self.warnings.append({
                                    "rule_id": "SMD-4",
                                    "description": "Activity missing blocks",
                                    "warnings": [f"Activity {activity.name} does not have blocks"]
                                })
                                continue
                            if state.name in activity.blocks:
                                found_behavior = True
                    if not found_behavior:
                        errors.append(f"State {state.name} in state machine {package.name} has no corresponding behavior in Activity Diagram")
        return len(errors) == 0, errors

    def validate_pd_constraints(self):
        """PD-1: Constraints must match design-block attributes and be valid/parsable."""
        errors = []
        for package in self.package_tree:
            if package.type == "parametric":
                if not hasattr(package, 'constraints'):
                    self.warnings.append({
                        "rule_id": "PD-1",
                        "description": "Parametric diagram missing constraints",
                        "warnings": [f"Parametric diagram {package.name} does not have constraints"]
                    })
                    continue
                for constraint in package.constraints:
                    if not hasattr(constraint, 'attribute'):
                        self.warnings.append({
                            "rule_id": "PD-1",
                            "description": "Constraint missing attribute",
                            "warnings": [f"Constraint in parametric diagram {package.name} does not have attribute"]
                        })
                        continue
                    found_attribute = False
                    for block in self.package_tree:
                        if block.type in ["block", "part"]:
                            if not hasattr(block, 'attributes'):
                                self.warnings.append({
                                    "rule_id": "PD-1",
                                    "description": "Block/Part missing attributes",
                                    "warnings": [f"{block.type} {block.name} does not have attributes"]
                                })
                                continue
                            for attr in block.attributes:
                                if constraint.attribute == attr.name:
                                    found_attribute = True
                    if not found_attribute:
                        errors.append(f"Constraint {constraint.name} in parametric diagram {package.name} has no matching attribute in BDD/IBD")
        return len(errors) == 0, errors 
    

class StatefulRule:
    """
    Stateful rule model used to track rule statistics and state.
    """
    def __init__(self, 
                 rule_id: str,
                 expression: str,
                 description: str,
                 total_checks: int = 0,
                 total_failures: int = 0,
                 checks_since_last_failure: int = 0,
                 failure_rate: float = 0.0,
                 is_breaker_open: bool = False,
                 manual_corrections: int = 0,
                 last_check_time = None):
        """
        Initialize a stateful rule.
        
        Args:
            rule_id: rule ID
            expression: rule expression
            description: rule description
            total_checks: total checks
            total_failures: total failures
            checks_since_last_failure: consecutive checks since last failure
            failure_rate: failure rate
            is_breaker_open: whether the circuit breaker is open
            manual_corrections: number of manual corrections
            last_check_time: last check time
        """
        self.rule_id = rule_id
        self.expression = expression
        self.description = description
        self.total_checks = total_checks
        self.total_failures = total_failures
        self.checks_since_last_failure = checks_since_last_failure
        self.failure_rate = failure_rate
        self.is_breaker_open = is_breaker_open
        self.manual_corrections = manual_corrections
        self.last_check_time = last_check_time or datetime.now()

    def to_dict(self) -> dict:
        """Convert to a dict."""
        return {
            'rule_id': self.rule_id,
            'expression': self.expression,
            'description': self.description,
            'total_checks': self.total_checks,
            'total_failures': self.total_failures,
            'checks_since_last_failure': self.checks_since_last_failure,
            'failure_rate': self.failure_rate,
            'is_breaker_open': self.is_breaker_open,
            'manual_corrections': self.manual_corrections,
            'last_check_time': self.last_check_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StatefulRule':
        """Create from a dict."""
        data['last_check_time'] = datetime.fromisoformat(data['last_check_time'])
        return cls(**data)

class RuleStateManager:
    """
    Rule state manager used to manage state for all rules.
    """
    def __init__(self, 
                 failure_rate_threshold: float = 0.1,
                 consecutive_checks_threshold: int = 10,
                 manual_corrections_threshold: int = 3,
                 state_file: str = "AutoMBSE/out/rule_states.json"):
        """
        Initialize the rule state manager.
        
        Args:
            failure_rate_threshold: failure rate threshold
            consecutive_checks_threshold: consecutive checks threshold
            manual_corrections_threshold: manual corrections threshold
            state_file: state file path
        """
        self.failure_rate_threshold = failure_rate_threshold
        self.consecutive_checks_threshold = consecutive_checks_threshold
        self.manual_corrections_threshold = manual_corrections_threshold
        state_file = os.environ.get("AUTOMBSE_RULE_STATE_FILE", state_file)
        self.state_file = state_file
        self.rules: dict[str, StatefulRule] = {}
        self.load_states()

    def load_states(self):
        """Load rule states from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.rules = {
                    rule_id: StatefulRule.from_dict(rule_data)
                    for rule_id, rule_data in data.items()
                }

    def save_states(self):
        """Save rule states to file."""
        data = {
            rule_id: rule.to_dict()
            for rule_id, rule in self.rules.items()
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_rule(self, rule: StatefulRule):
        """Add a new rule."""
        self.rules[rule.rule_id] = rule
        self.save_states()

    def update_rule_state(self, 
                         rule_id: str, 
                         check_result: bool,
                         manual_correction: bool = False):
        """
        Update rule state.
        
        Args:
            rule_id: rule ID
            check_result: check result
            manual_correction: whether this is a manual correction
        """
        if rule_id not in self.rules:
            return

        rule = self.rules[rule_id]
        rule.total_checks += 1
        rule.last_check_time = datetime.now()

        if not check_result:
            rule.total_failures += 1
            rule.checks_since_last_failure = 0
        else:
            rule.checks_since_last_failure += 1

        if manual_correction:
            rule.manual_corrections += 1

        # Update failure rate
        rule.failure_rate = rule.total_failures / rule.total_checks

        # Check whether to trigger the circuit breaker
        if (rule.failure_rate < self.failure_rate_threshold and 
            rule.checks_since_last_failure > self.consecutive_checks_threshold):
            rule.is_breaker_open = True

        # Check whether manual corrections reach the threshold
        if rule.manual_corrections >= self.manual_corrections_threshold:
            rule.is_breaker_open = False

        self.save_states()

    def get_active_rules(self) -> list[StatefulRule]:
        """Get currently active rules (circuit breaker is closed)."""
        return [rule for rule in self.rules.values() if not rule.is_breaker_open]

    def get_rule_stats(self) -> dict:
        """Get rule statistics."""
        return {
            'total_rules': len(self.rules),
            'active_rules': len(self.get_active_rules()),
            'rules_by_failure_rate': {
                'low': len([r for r in self.rules.values() if r.failure_rate < 0.1]),
                'medium': len([r for r in self.rules.values() if 0.1 <= r.failure_rate < 0.3]),
                'high': len([r for r in self.rules.values() if r.failure_rate >= 0.3])
            },
            'rules_by_corrections': {
                'low': len([r for r in self.rules.values() if r.manual_corrections < 3]),
                'medium': len([r for r in self.rules.values() if 3 <= r.manual_corrections < 5]),
                'high': len([r for r in self.rules.values() if r.manual_corrections >= 5])
            }
        }

class Package:
    """Package model used to represent a SysML package tree."""
    def __init__(self, **kwargs):
        # Set default attributes
        self.name = None
        self.type = None
        self.level = 0
        self.project_id = 0
        self.view_type = None
        self.children = []
        self.belongsto = None
        self.generalization = None
        self.composition = None
        self.attributes = []
        self.behaviors = []
        self.ports = []
        self.connections = []
        self.satisfies = []
        self.constraints = []
        self.states = []
        self.transitions = []
        self.actions = []
        self.events = []
        self.signal_types = []
        
        # Update attribute values
        for key, value in kwargs.items():
            if key == 'children' and value:
                self.children = [Package(**child) for child in value]
            else:
                setattr(self, key, value)

    def __str__(self):
        """Return a short string representation."""
        return f"Package(name={self.name}, type={self.type}, level={self.level})"

    def __repr__(self):
        """Return a detailed string representation."""
        return f"Package(name={self.name}, type={self.type}, level={self.level}, children={len(self.children)})"

class Rules(BDDRules, CrossRules):
    """
    Rule checker class that combines BDDRules and CrossRules and maintains rule state.
    """
    def __init__(self, package_tree, failure_rate_threshold = 0.01, consecutive_checks_threshold = 100, manual_corrections_threshold = 10):
        """
        Initialize rule checker.
        
        Args:
            package_tree: package tree (dict or objects)
        """
        # Convert dict-based tree into object-based tree
        if isinstance(package_tree, dict):
            self.package_tree = [Package(**pkg) for pkg in package_tree]
        elif isinstance(package_tree, list):
            if package_tree and isinstance(package_tree[0], dict):
                self.package_tree = [Package(**pkg) for pkg in package_tree]
            else:
                self.package_tree = package_tree
        else:
            self.package_tree = package_tree

        # Initialize base classes
        BDDRules.__init__(self, self.package_tree)
        CrossRules.__init__(self, self.package_tree)
        
        # Initialize rule state manager
        self.rule_manager = RuleStateManager(
            failure_rate_threshold=failure_rate_threshold,
            consecutive_checks_threshold=consecutive_checks_threshold,
            manual_corrections_threshold=manual_corrections_threshold
        )
        
        # Initialize stateful rules
        self._initialize_stateful_rules()
        
        # Merge rule lists
        self.all_rules = {
            "bdd": self.erules + self.wrules,  # BDD rules
            "cross": self.rules,  # cross-view rules
        }
        
        # Merge error/warning lists
        self.all_errors = []
        self.all_warnings = []

    def _initialize_stateful_rules(self):
        """Initialize state for all rules."""
        # Initialize BDD rules
        for rule in self.erules + self.wrules:
            stateful_rule = StatefulRule(
                rule_id=rule['rule_id'],
                expression=str(rule['check_function']),
                description=rule['description']
            )
            self.rule_manager.add_rule(stateful_rule)
        
        # Initialize cross-view rules
        for rule in self.rules:
            stateful_rule = StatefulRule(
                rule_id=rule['rule_id'],
                expression=str(rule['check_function']),
                description=rule['description']
            )
            self.rule_manager.add_rule(stateful_rule)

    def validate_all(self):
        """
        Execute all rule checks and update rule state.
        
        Returns:
            tuple: (errors, warnings) all errors and warnings
        """
        # Get active rules
        active_rules = self.rule_manager.get_active_rules()
        active_rule_ids = {rule.rule_id for rule in active_rules}
        
        # Run BDD rule checks
        bdd_errors, bdd_warnings = self._validate_bdd_rules(active_rule_ids)
        
        # Run cross-view rule checks
        cross_errors, cross_warnings = self._validate_cross_rules(active_rule_ids)
        
        # Merge results
        self.all_errors = bdd_errors + cross_errors
        self.all_warnings = bdd_warnings + cross_warnings
        
        return self.all_errors, self.all_warnings

    def _validate_bdd_rules(self, active_rule_ids):
        """Run BDD rule checks and update state."""
        errors = []
        warnings = []
        
        for rule in self.erules + self.wrules:
            if rule['rule_id'] not in active_rule_ids:
                continue
                
            valid, results = rule['check_function'](self.package_tree)
            
            # Update rule state
            self.rule_manager.update_rule_state(
                rule['rule_id'],
                check_result=valid
            )
            
            if not valid:
                if rule in self.erules:
                    errors.append({
                        "rule_id": rule['rule_id'],
                        "description": rule['description'],
                        "errors": results
                    })
                else:
                    warnings.append({
                        "rule_id": rule['rule_id'],
                        "description": rule['description'],
                        "warnings": results
                    })
        
        return errors, warnings

    def _validate_cross_rules(self, active_rule_ids):
        """Run cross-view rule checks and update state."""
        errors = []
        warnings = []
        
        for rule in self.rules:
            if rule['rule_id'] not in active_rule_ids:
                continue
                
            valid, results = rule['check_function']()
            
            # Update rule state
            self.rule_manager.update_rule_state(
                rule['rule_id'],
                check_result=valid
            )
            
            if not valid:
                errors.append({
                    "rule_id": rule['rule_id'],
                    "description": rule['description'],
                    "errors": results
                })
        
        return errors, warnings

    def validate_by_type(self, rule_type):
        """
        Run rule checks by type.
        
        Args:
            rule_type (str): rule type: 'bdd', 'cross', or 'all'
            
        Returns:
            tuple: (errors, warnings) errors and warnings for the selected type
        """
        active_rule_ids = {rule.rule_id for rule in self.rule_manager.get_active_rules()}
        
        if rule_type == 'bdd':
            return self._validate_bdd_rules(active_rule_ids)
        elif rule_type == 'cross':
            return self._validate_cross_rules(active_rule_ids)
        elif rule_type == 'all':
            return self.validate_all()
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    def get_rule_info(self, rule_type=None):
        """
        Get rule info, including state.
        
        Args:
            rule_type (str, optional): rule type: 'bdd', 'cross', or 'all'
            
        Returns:
            dict: rule info
        """
        rule_info = {}
        
        if rule_type in ['bdd', 'all']:
            rule_info['bdd'] = {
                'error_rules': [{
                    'rule_id': rule['rule_id'],
                    'description': rule['description'],
                    'state': self.rule_manager.rules[rule['rule_id']].to_dict()
                } for rule in self.erules],
                'warning_rules': [{
                    'rule_id': rule['rule_id'],
                    'description': rule['description'],
                    'state': self.rule_manager.rules[rule['rule_id']].to_dict()
                } for rule in self.wrules]
            }
            
        if rule_type in ['cross', 'all']:
            rule_info['cross'] = [{
                'rule_id': rule['rule_id'],
                'description': rule['description'],
                'state': self.rule_manager.rules[rule['rule_id']].to_dict()
            } for rule in self.rules]
            
        return rule_info

    def get_validation_summary(self):
        """
        Get a summary of validation results, including rule statistics.
        
        Returns:
            dict: validation summary
        """
        rule_stats = self.rule_manager.get_rule_stats()
        
        return {
            'total_errors': len(self.all_errors),
            'total_warnings': len(self.all_warnings),
            'bdd_errors': len([e for e in self.all_errors if e['rule_id'].startswith('BDD')]),
            'bdd_warnings': len([w for w in self.all_warnings if w['rule_id'].startswith('BDD')]),
            'cross_errors': len([e for e in self.all_errors if not e['rule_id'].startswith('BDD')]),
            'cross_warnings': len([w for w in self.all_warnings if not w['rule_id'].startswith('BDD')]),
            'rule_stats': rule_stats
        }

    def record_manual_correction(self, rule_id: str):
        """
        Record a manual correction.
        
        Args:
            rule_id: rule ID
        """
        if rule_id in self.rule_manager.rules:
            self.rule_manager.update_rule_state(
                rule_id,
                check_result=True,
                manual_correction=True
            )
