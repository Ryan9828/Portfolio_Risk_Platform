"""Announcement intelligence — LLM-extracted risk signals from ASX announcements.

ingest    pull announcement headlines per ticker from the public ASX feed
extract   Claude turns each headline into a typed signal (event type, sentiment,
          materiality) with per-call cost and latency recorded
events    event study linking signals to abnormal returns and volatility regimes
evals     golden-set evaluation of the extraction step (precision/recall)
"""
