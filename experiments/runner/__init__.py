"""
Experiment runner.

ExperimentRunner (runner.py) drives the experiment sweep: for each
Configuration x repetition in an Experiment, resolves the AI provider
if needed, verifies the Scenario's dataset checksum, invokes the
corresponding strategy (experiments/strategies/), and applies the
common measurement layer (experiments/metrics/) uniformly to produce
a Run with an attached Result. This is the "invoke CONVERGE (or a
baseline) without modifying it" harness referred to throughout this
package's docstrings. Purely in-memory -- no persistence in this
phase.
"""
