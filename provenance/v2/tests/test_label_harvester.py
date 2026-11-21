"""Unit tests for backward label harvesters."""

import pytest
from provenance.v2.label_harvester import (
    harvest_gujarati_labels,
    harvest_hebrew_labels,
    harvest_pragmatic_labels,
    harvest_vietnamese_labels,
)


def test_hebrew_harvester() -> None:
    # Male pronoun
    prof, req, forb = harvest_hebrew_labels("אתה הולך לשוק עכשיו?")
    assert prof.listener_gender == "male"
    assert "אתה" in req
    assert "את" in forb

    # Female pronoun
    prof2, req2, forb2 = harvest_hebrew_labels("את הולכת הביתה היום?")
    assert prof2.listener_gender == "female"
    assert "את" in req2
    assert "אתה" in forb2

    # Direct object marker 'את ה...' should not be harvested as female listener
    prof3, req3, forb3 = harvest_hebrew_labels("ראיתי את הכלב בחצר.")
    assert prof3.listener_gender is None


def test_gujarati_harvester() -> None:
    # L1 intimate
    prof_l1, req_l1, forb_l1 = harvest_gujarati_labels("તું ક્યારે ઘરે આવીશ?")
    assert prof_l1.register_tier == "intimate"
    assert "તું" in req_l1
    assert "તમે" in forb_l1

    # L2 polite
    prof_l2, req_l2, forb_l2 = harvest_gujarati_labels("તમે ક્યારે ઘરે આવશો?")
    assert prof_l2.register_tier == "polite"
    assert "તમે" in req_l2
    assert "તું" in forb_l2

    # L3 honorific
    prof_l3, req_l3, forb_l3 = harvest_gujarati_labels("આપ ક્યારે પધારશો?")
    assert prof_l3.register_tier == "honorific"
    assert "આપ" in req_l3
    assert "તમે" in forb_l3


def test_vietnamese_harvester() -> None:
    # Kinship pair
    prof_kin, req_kin, forb_kin = harvest_vietnamese_labels("Anh muốn hỏi em một việc.")
    assert prof_kin.social_relationship == "anh_em"
    assert "anh" in req_kin
    assert "em" in req_kin

    # Southern dialect marker
    prof_so, req_so, forb_so = harvest_vietnamese_labels("Lát nữa em có qua bển hổng?")
    assert prof_so.regional_dialect == "vi_south"
    assert "hổng" in req_so
    assert "không" in forb_so

    # Northern dialect marker
    prof_no, req_no, forb_no = harvest_vietnamese_labels("Hôm nay tôi vào đó một lát nhé.")
    assert prof_no.regional_dialect == "vi_north"
    assert "vào" in req_no
    assert "vô" in forb_no
