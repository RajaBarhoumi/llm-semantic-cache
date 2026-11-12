"""Normalization and entity guard."""

from app.cache.guards import anchors, compatible, normalize


def test_normalize():
    assert normalize("  Where is   my ORDER?? ") == "where is my order"
    assert normalize("Don’t") == "don't"


def test_numbers_and_ids_must_match():
    assert not compatible("invoice for order 1001", "invoice for order 1002")
    assert not compatible("status of ticket INV-2031", "status of ticket INV-2032")
    assert compatible("invoice for order 1001", "can I get the invoice of order #1001")


def test_names_must_match_case_insensitively():
    assert not compatible("Do you ship to Canada?", "Do you ship to France?")
    assert compatible("Do you ship to Canada?", "do you deliver to canada")
    assert compatible("Can I pay with PayPal?", "do you accept paypal")


def test_emails_must_match():
    assert not compatible("send it to a@b.com", "send it to c@d.com")


def test_sentence_start_is_not_a_name():
    assert anchors("Hi. Where is my parcel?") == set()
    assert anchors("I can't log in") == set()
