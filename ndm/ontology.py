#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from random import randint

debug = False
# debug = True


class Arguments:
    def __init__(self, n_arguments_per_slot):
        self.n_arguments_per_slot = n_arguments_per_slot
        self.value2argument = {}
        self.argument2value = {}

    def __str__(self):
        return str(self.argument2value)

    def add(self, value, argument):
        self.value2argument[value] = argument
        self.argument2value[argument] = value

    def get_argument(self, value, default_slot):
        if value in self.value2argument:
            return self.value2argument[value]
        else:
            # I must add the argument and the return it
            if self.n_arguments_per_slot > 1:
                for i in range(0, self.n_arguments_per_slot * 3):
                    n = randint(0, self.n_arguments_per_slot - 1)
                    ARGUMENT = default_slot + '_' + str(n)

                    if ARGUMENT in self.argument2value:
                        continue
                    else:
                        self.add(value, ARGUMENT)
                        return ARGUMENT

                # overwrite an existing argument !!!
                print('-' * 120)
                print('WARNING:   overwriting an existing argument ARGUMENT = {a}, VALUE = {v}'.format(a=ARGUMENT, v=value))
                print('ARGUMENTS:', self.argument2value)
                print()
                self.add(value, ARGUMENT)
            else:
                # there is only one argument per slot
                ARGUMENT = default_slot + '_X'
                self.add(value, ARGUMENT)
            return ARGUMENT


class Ontology:
    def __init__(self, ontology_fn, database_fn):
        self.load_ontology(ontology_fn, database_fn)

    def load_ontology(self, ontology_fn, database_fn):
        """Load ontology - a json file. And extend it with values from teh database.

        :param ontology_fn: a name of the ontology json file
        :param database_fn: a name of the database json file
        """
        with open(ontology_fn) as f:
            ontology = json.load(f)

        with open(database_fn) as f:
            database = json.load(f)

        # slot-value-form mapping
        self.svf = defaultdict(lambda: defaultdict(set))
        # form-value-slot mapping
        self.fvs = defaultdict(lambda: defaultdict(set))
        # form-value mapping
        self.fv = {}
        # form-slot mapping
        self.fs = {}
        for slot in ontology['informable']:
            for value in ontology['informable'][slot]:
                self.add_onto_entry(slot, value, value)

        for entity in database:
            for slot in entity:
                value = entity[slot]

                self.add_onto_entry(slot, value, value)

        # HACK: add one actra sufrace form alternative
        self.add_onto_entry('PRICERANGE', 'moderate', 'moderately')

    def add_onto_entry(self, slot, value, form):
        self.svf[slot][value].add(tuple(form.lower().split()))
        self.fvs[tuple(form.lower().split())][value].add(slot)
        self.fv[tuple(form.lower().split())] = value
        self.fs[tuple(form.lower().split())] = slot.upper()

    def abstract_utterance_helper(self, init_i, utterance, history_arguments):
        for i in range(init_i, len(utterance)):
            for j in range(len(utterance), i, -1):
                slice = tuple(utterance[i:j])

                # print('s', slice)
                if slice in self.fvs:
                    value = self.fv[slice]
                    # print('f', value)

                    # skip this common false positive
                    if i > 5:
                        # print(utterance[i-1:j+1])
                        if utterance[i - 1:j + 1] == ['can', 'ask', 'for']:
                            continue

                    ARGUMENT = [history_arguments.get_argument(value, self.fs[slice])]

                    return i + 1, list(utterance[:i]) + ARGUMENT + list(utterance[j:]), True

        return len(utterance), utterance, False

    def abstract_utterance(self, utterance, history_arguments):
        abs_utt = utterance

        changed = True
        i = 0
        while changed:
            i, abs_utt, changed = self.abstract_utterance_helper(i, abs_utt, history_arguments)
            # if changed:
            #     print('au', abs_utt)
        return abs_utt

    def abstract_target(self, target, history_arguments, mode):
        if mode == 'tracker':
            return self.abstract_utterance(target, history_arguments)
        else:  # mode == 'e2e'
            return self.abstract_utterance(target, history_arguments)

    def abstract(self, examples, mode):
        abstract_examples = []
        arguments = []

        for history, target in examples:
            if debug:
                print('=' * 120)
            abstract_history = []
            history_arguments = Arguments(n_arguments_per_slot=5)
            for utterance in history:
                abs_utt = self.abstract_utterance(utterance, history_arguments)
                abstract_history.append(abs_utt)

                if debug:
                    print('U    ', utterance)
                    print('AbsU ', abs_utt)
                    print('Args ', history_arguments)
                    print()

            if mode == 'tracker':
                abs_tgt = self.abstract_utterance(target, history_arguments)
            else:  # mode == 'e2e'
                target_arguments = Arguments(n_arguments_per_slot=1)
                abs_tgt = self.abstract_utterance(target, target_arguments)
            if debug:
                print('T    ', target)
                print('AbsT ', abs_tgt)
                if mode == 'tracker':
                    print('Args ', history_arguments)
                else:  # mode == 'e2e'
                    print('Args ', target_arguments)
                print()

            abstract_examples.append([abstract_history, abs_tgt])

        return abstract_examples, arguments
