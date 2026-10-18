from app.presentation import chase_verdict


def test_verdict_always_names_the_side_or_describes_a_close_chase():
    assert chase_verdict(0.70) == "Chasing side favoured"
    assert chase_verdict(0.50) == "Close chase · keep an eye"
    assert chase_verdict(0.30) == "Defending side favoured"
    assert chase_verdict(None) == "Not enough historical support"
    assert chase_verdict(1.1) == "Not enough historical support"
