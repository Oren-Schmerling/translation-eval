#!/usr/bin/env python3
"""Generate the curated evaluation sheets for Hebrew, Gujarati, and Vietnamese.

The script deliberately depends only on the Python standard library.  It
starts from 36 English base sentences (18 family / 18 daily-life) and expands
each one through a deterministic pragmatic matrix into exactly:

    data/templates/he_curation.csv   192 data rows
    data/templates/gu_curation.csv   288 data rows
    data/templates/vi_curation.csv   240 data rows

Run directly with:

    python3 scripts/generate_sheets_template.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "templates"

HEADER = [
    "item_id",
    "variant_id",
    "source_en",
    "lang",
    "track",
    "bucket",
    "is_cued_source",
    "formality",
    "speaker_gender",
    "listener_gender",
    "dialect",
    "pronoun_tier",
    "kinship_pair",
    "relationship",
    "conflict_formality",
    "conflict_speaker_gender",
    "conflict_listener_gender",
    "conflict_dialect",
    "gold_reference",
]

DOMAIN_FAMILY = "family_and_relatives"
DOMAIN_CULTURE = "culture_and_daily_life"


def _item(
    item_id: str,
    domain: str,
    bucket: str,
    en: str,
    *,
    he: str,
    he_sp: Sequence[str],
    he_voc: Dict[str, str] | None = None,
    gu: str,
    gu_sp: Sequence[str],
    gu_voc: Dict[str, str] | None = None,
    vi: str,
) -> dict:
    return {
        "id": item_id,
        "domain": domain,
        "bucket": bucket,
        "en": en,
        "he": he,
        "he_sp": {"male": he_sp[0], "female": he_sp[1]},
        "he_voc": he_voc or {},
        "gu": gu,
        "gu_sp": {"male": gu_sp[0], "female": gu_sp[1]},
        "gu_voc": gu_voc or {},
        "vi": vi,
    }


ITEMS: List[dict] = [
    # ------------------------------------------------------------------
    # Family and relatives -- bucket A (second-person listener focus)
    # ------------------------------------------------------------------
    _item(
        "fam_a_01", DOMAIN_FAMILY, "A",
        "You look tired today; are you okay?",
        he="{formal}אני {sp} שאתה עייף היום, אתה בסדר?",
        he_sp=("חושש", "חוששת"),
        gu="હું {sp} છું; {pd} આજે આરામ કરવો જોઈએ.",
        gu_sp=("ચિંતિત થયો", "ચિંતિત થઈ"),
        vi="Hôm nay {listen} trông mệt quá, {self} lo cho {listen}, {listen} có ổn không?",
    ),
    _item(
        "fam_a_02", DOMAIN_FAMILY, "A",
        "Please call your mother when you get home.",
        he="{formal}אני {sp} שתקרא לאמא שלך כשתחזור הביתה.",
        he_sp=("מבקש", "מבקשת"),
        gu="{pd} ઘરે પહોંચીને મમ્મીને ફોન કરવો જોઈએ; હું {sp} છું.",
        gu_sp=("કહેતો", "કહેતી"),
        vi="{listen} nhớ gọi cho mẹ khi về đến nhà, {self} dặn {listen} đấy.",
    ),
    _item(
        "fam_a_03", DOMAIN_FAMILY, "A",
        "You should listen to your grandmother's advice.",
        he="{formal}אני {sp} לך להקשיב לעצת סבתא שלך.",
        he_sp=("ממליץ", "ממליצה"),
        gu="{pd} દાદીની સલાહ સાંભળવી જોઈએ, હું {sp} છું.",
        gu_sp=("સલાહ આપતો", "સલાહ આપતી"),
        vi="{self} khuyên {listen} nên nghe lời khuyên của bà.",
    ),
    _item(
        "fam_a_04", DOMAIN_FAMILY, "A",
        "Do you want to visit your cousins this weekend?",
        he="{formal}אני {sp} אם תרצה לבקר את הדודים שלך בסוף השבוע.",
        he_sp=("סקרן", "סקרנית"),
        gu="{pd} આ સપ્તાહે પિતરાઈ ભાઈ-બહેનને મળવું છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        vi="{listen} có muốn thăm anh chị em họ với {self} vào cuối tuần này không?",
    ),
    _item(
        "fam_a_05", DOMAIN_FAMILY, "A",
        "You forgot to wish your father a happy birthday.",
        he="{formal}אני {sp} ששכחת לאחל לאבא שלך יום הולדת שמח.",
        he_sp=("מופתע", "מופתעת"),
        gu="{pd} યાદ રહ્યું કે પપ્પાને જન્મદિવસની શુભેચ્છા આપવાની બાકી છે? હું {sp} છું.",
        gu_sp=("યાદ કરાવતો", "યાદ કરાવતી"),
        vi="{listen} quên chúc bố sinh nhật vui vẻ rồi, {self} nhắc {listen} nhé.",
    ),
    _item(
        "fam_a_06", DOMAIN_FAMILY, "A",
        "Can you help your younger brother with his homework?",
        he="{formal}אני {sp} שתעזור לאחיך הקטן בשיעורי הבית.",
        he_sp=("מבקש", "מבקשת"),
        gu="{pd} નાના ભાઈને ગૃહકાર્યમાં મદદ કરવી છે? હું {sp} છું.",
        gu_sp=("વિનંતી કરતો", "વિનંતી કરતી"),
        vi="{listen} giúp em trai làm bài tập được không? {self} nhờ {listen} đấy.",
    ),
    # ------------------------------------------------------------------
    # Family and relatives -- bucket B (first-person speaker focus)
    # ------------------------------------------------------------------
    _item(
        "fam_b_01", DOMAIN_FAMILY, "B",
        "I am ready to help you with the children.",
        he="{formal}אני {sp} לעזור לך עם הילדים.",
        he_sp=("מוכן", "מוכנה"),
        gu="હું {sp} છું અને {pd} બાળકો સાથે મદદ કરવા તૈયાર છું.",
        gu_sp=("મદદ કરવા તૈયાર થયો", "મદદ કરવા તૈયાર થઈ"),
        vi="{self} sẵn sàng giúp {listen} trông bọn trẻ.",
    ),
    _item(
        "fam_b_02", DOMAIN_FAMILY, "B",
        "I miss my family when I travel.",
        he="{formal}אני {sp} את המשפחה שלי כשאני נוסע.",
        he_sp=("מתגעגע", "מתגעגעת"),
        gu="જ્યારે હું મુસાફરી કરું છું ત્યારે મને મારા પરિવારની યાદ આવે છે; હું {sp} છું.",
        gu_sp=("ઉદાસ થયો", "ઉદાસ થઈ"),
        vi="{self} nhớ gia đình khi đi xa.",
    ),
    _item(
        "fam_b_03", DOMAIN_FAMILY, "B",
        "I will bring dinner to my parents tonight.",
        he="{formal}אני {sp} להביא ארוחת ערב להורים שלי הערב.",
        he_sp=("מתכוון", "מתכוונת"),
        gu="હું આજે સાંજે મારા માતા-પિતા માટે રાત્રિભોજન લઈ જવા {sp} છું.",
        gu_sp=("નીકળ્યો", "નીકળી"),
        vi="{self} sẽ mang bữa tối đến cho bố mẹ tối nay.",
    ),
    _item(
        "fam_b_04", DOMAIN_FAMILY, "B",
        "I am proud of my daughter.",
        he="{formal}אני {sp} גאווה בבת שלי.",
        he_sp=("מרגיש", "מרגישה"),
        gu="હું મારી દીકરી પર {sp} છું.",
        gu_sp=("ગર્વિત થયો", "ગર્વિત થઈ"),
        vi="{self} tự hào về con gái của {self}.",
    ),
    _item(
        "fam_b_05", DOMAIN_FAMILY, "B",
        "I need to call my uncle tomorrow.",
        he="{formal}אני {sp} להתקשר לדוד שלי מחר.",
        he_sp=("צריך", "צריכה"),
        gu="મારે કાલે મારા કાકાને ફોન કરવો છે; હું {sp} છું.",
        gu_sp=("યાદ કરતો", "યાદ કરતી"),
        vi="{self} cần gọi cho chú của {self} vào ngày mai.",
    ),
    _item(
        "fam_b_06", DOMAIN_FAMILY, "B",
        "I bought a gift for my sister.",
        he="{formal}אני {sp} מהמתנה שקניתי לאחותי.",
        he_sp=("נרגש", "נרגשת"),
        gu="મેં મારી બહેન માટે ભેટ ખરીદી છે અને હું {sp} છું.",
        gu_sp=("ખુશ થયો", "ખુશ થઈ"),
        vi="{self} đã mua quà cho chị gái của {self}.",
    ),
    # ------------------------------------------------------------------
    # Family and relatives -- bucket C (cued dialogic, vocatives)
    # ------------------------------------------------------------------
    _item(
        "fam_c_01", DOMAIN_FAMILY, "C",
        "Mom, I am ready to help you in the kitchen.",
        he="{formal}{voc}, אני {sp} לעזור לך במטבח.",
        he_sp=("מוכן", "מוכנה"),
        he_voc={"male": "אבא", "female": "אמא"},
        gu="{voc}, હું {sp} છું અને {pd} રસોડામાં મદદ કરવા તૈયાર છું.",
        gu_sp=("તૈયાર થયો", "તૈયાર થઈ"),
        gu_voc={"male": "પપ્પા", "female": "મમ્મી"},
        vi="{listen} ơi, {self} sẵn sàng giúp {listen} trong bếp.",
    ),
    _item(
        "fam_c_02", DOMAIN_FAMILY, "C",
        "Grandfather, can you tell me about your childhood?",
        he="{formal}{voc}, אני {sp} לשמוע על הילדות שלך.",
        he_sp=("סקרן", "סקרנית"),
        he_voc={"male": "סבא", "female": "סבתא"},
        gu="{voc}, {pd} પોતાના બાળપણ વિશે કહેવું છે? હું {sp} છું.",
        gu_sp=("ઉત્સુક થયો", "ઉત્સુક થઈ"),
        gu_voc={"male": "દાદા", "female": "દાદી"},
        vi="{listen} ơi, {self} muốn nghe {listen} kể về tuổi thơ của {listen}.",
    ),
    _item(
        "fam_c_03", DOMAIN_FAMILY, "C",
        "My dear brother, I need your advice.",
        he="{formal}{voc}, אני {sp} בעצתך.",
        he_sp=("זקוק", "זקוקה"),
        he_voc={"male": "אחי", "female": "אחותי"},
        gu="{voc}, મને {pd} સલાહની જરૂર છે; હું {sp} છું.",
        gu_sp=("ચિંતિત થયો", "ચિંતિત થઈ"),
        gu_voc={"male": "ભાઈ", "female": "બહેન"},
        vi="{listen} ơi, {self} cần lời khuyên của {listen}.",
    ),
    _item(
        "fam_c_04", DOMAIN_FAMILY, "C",
        "Aunt, I will visit you on Friday.",
        he="{formal}{voc}, אני {sp} לבקר אותך ביום שישי.",
        he_sp=("מתכוון", "מתכוונת"),
        he_voc={"male": "דוד", "female": "דודה"},
        gu="{voc}, હું શુક્રવારે {pd} મળવા આવીશ; હું {sp} છું.",
        gu_sp=("ઉત્સાહિત થયો", "ઉત્સાહિત થઈ"),
        gu_voc={"male": "કાકા", "female": "કાકી"},
        vi="{listen} ơi, {self} sẽ đến thăm {listen} vào thứ Sáu.",
    ),
    _item(
        "fam_c_05", DOMAIN_FAMILY, "C",
        "Son, I am proud of you.",
        he="{formal}{voc}, אני {sp} גאווה בך.",
        he_sp=("מרגיש", "מרגישה"),
        he_voc={"male": "בני", "female": "בתי"},
        gu="{voc}, {pd} જોઈને હું {sp} છું.",
        gu_sp=("ગર્વ અનુભવતો", "ગર્વ અનુભવતી"),
        gu_voc={"male": "દીકરા", "female": "દીકરી"},
        vi="{listen} ơi, {self} tự hào về {listen}.",
    ),
    _item(
        "fam_c_06", DOMAIN_FAMILY, "C",
        "My sister, I miss you very much.",
        he="{formal}{voc}, אני {sp} אותך מאוד.",
        he_sp=("מתגעגע", "מתגעגעת"),
        he_voc={"male": "אחי", "female": "אחותי"},
        gu="{voc}, મને {pd} ખૂબ યાદ આવે છે; હું {sp} છું.",
        gu_sp=("ઉદાસ થયો", "ઉદાસ થઈ"),
        gu_voc={"male": "ભાઈ", "female": "બહેન"},
        vi="{listen} ơi, {self} nhớ {listen} rất nhiều.",
    ),
    # ------------------------------------------------------------------
    # Culture and daily life -- bucket A (second-person focus)
    # ------------------------------------------------------------------
    _item(
        "cul_a_01", DOMAIN_CULTURE, "A",
        "You should drink tea with milk in the morning.",
        he="{formal}אני {sp} לך לשתות תה עם חלב בבוקר.",
        he_sp=("ממליץ", "ממליצה"),
        gu="હું {sp} છું; {pd} સવારે દૂધવાળી ચા પીવી જોઈએ.",
        gu_sp=("સૂચન કરતો", "સૂચન કરતી"),
        vi="{self} khuyên {listen} nên uống trà với sữa vào buổi sáng.",
    ),
    _item(
        "cul_a_02", DOMAIN_CULTURE, "A",
        "Do you know where the market is?",
        he="{formal}אני {sp} אם אתה יודע איפה השוק.",
        he_sp=("סקרן", "סקרנית"),
        gu="{pd} ખબર છે બજાર ક્યાં છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        vi="{listen} có biết chợ ở đâu không? {self} muốn hỏi {listen}.",
    ),
    _item(
        "cul_a_03", DOMAIN_CULTURE, "A",
        "You need to wear a warm coat today.",
        he="{formal}אני {sp} שתחבש מעיל חם היום.",
        he_sp=("מבקש", "מבקשת"),
        gu="{pd} આજે ગરમ કોટ પહેરવો જોઈએ; હું {sp} છું.",
        gu_sp=("કહેતો", "કહેતી"),
        vi="{self} nhắc {listen} cần mặc áo ấm hôm nay.",
    ),
    _item(
        "cul_a_04", DOMAIN_CULTURE, "A",
        "Can you buy some bread on your way home?",
        he="{formal}אני {sp} שתקנה קצת לחם בדרך הביתה.",
        he_sp=("מבקש", "מבקשת"),
        gu="{pd} ઘરે જતી વખતે થોડી રોટલી ખરીદવી છે? હું {sp} છું.",
        gu_sp=("વિનંતી કરતો", "વિનંતી કરતી"),
        vi="{listen} mua giúp {self} ít bánh mì trên đường về được không?",
    ),
    _item(
        "cul_a_05", DOMAIN_CULTURE, "A",
        "You are speaking too quickly; please slow down.",
        he="{formal}אני {sp} שאתה מדבר מהר מדי; בבקשה תאט.",
        he_sp=("מבחין", "מבחינה"),
        gu="{pd} ખૂબ ઝડપ છે, ધીમે બોલવું જોઈએ; હું {sp} છું.",
        gu_sp=("નોંધતો", "નોંધતી"),
        vi="{listen} nói nhanh quá, {self} xin {listen} chậm lại.",
    ),
    _item(
        "cul_a_06", DOMAIN_CULTURE, "A",
        "Do you want to watch the festival parade with us?",
        he="{formal}אני {sp} אם תרצה לראות את מצעד הפסטיבל איתנו.",
        he_sp=("סקרן", "סקרנית"),
        gu="{pd} આપણી સાથે ઉત્સવની પરેડ જોવી છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        vi="{listen} có muốn xem đoàn diễu hành lễ hội với {self} không?",
    ),
    # ------------------------------------------------------------------
    # Culture and daily life -- bucket B (first-person focus)
    # ------------------------------------------------------------------
    _item(
        "cul_b_01", DOMAIN_CULTURE, "B",
        "I drink coffee every morning.",
        he="{formal}אני {sp} לשתות קפה כל בוקר.",
        he_sp=("רגיל", "רגילה"),
        gu="હું દરરોજ સવારે કોફી પીવા {sp} છું.",
        gu_sp=("ટેવાયેલો", "ટેવાયેલી"),
        vi="{self} uống cà phê mỗi sáng.",
    ),
    _item(
        "cul_b_02", DOMAIN_CULTURE, "B",
        "I am learning to cook rice with vegetables.",
        he="{formal}אני {sp} לבשל אורז עם ירקות.",
        he_sp=("לומד", "לומדת"),
        gu="હું શાકભાજી સાથે ભાત બનાવવાનું {sp} છું.",
        gu_sp=("શીખતો", "શીખતી"),
        vi="{self} đang học nấu cơm với rau.",
    ),
    _item(
        "cul_b_03", DOMAIN_CULTURE, "B",
        "I will take the bus to the city center.",
        he="{formal}אני {sp} באוטובוס למרכז העיר.",
        he_sp=("נוסע", "נוסעת"),
        gu="હું શહેરના કેન્દ્ર સુધી બસમાં {sp} છું.",
        gu_sp=("જતો", "જતી"),
        vi="{self} sẽ đi xe buýt đến trung tâm thành phố.",
    ),
    _item(
        "cul_b_04", DOMAIN_CULTURE, "B",
        "I like the music at this festival.",
        he="{formal}אני {sp} את המוזיקה בפסטיבל הזה.",
        he_sp=("אוהב", "אוהבת"),
        gu="હું આ ઉત્સવના સંગીતનો આનંદ {sp} છું.",
        gu_sp=("લેતો", "લેતી"),
        vi="{self} thích âm nhạc ở lễ hội này.",
    ),
    _item(
        "cul_b_05", DOMAIN_CULTURE, "B",
        "I am tired after work today.",
        he="{formal}אני {sp} אחרי העבודה היום.",
        he_sp=("עייף", "עייפה"),
        gu="આજે કામ પછી હું {sp} છું.",
        gu_sp=("થાક્યો", "થાકી"),
        vi="{self} mệt sau giờ làm hôm nay.",
    ),
    _item(
        "cul_b_06", DOMAIN_CULTURE, "B",
        "I bought fresh fruit at the market.",
        he="{formal}אני {sp} שביעות רצון מהפירות שקניתי בשוק.",
        he_sp=("מרגיש", "מרגישה"),
        gu="મેં બજારમાંથી તાજા ફળ ખરીદ્યા છે અને હું {sp} છું.",
        gu_sp=("ખુશ થયો", "ખુશ થઈ"),
        vi="{self} đã mua trái cây tươi ở chợ.",
    ),
    # ------------------------------------------------------------------
    # Culture and daily life -- bucket C (cued dialogic, vocatives)
    # ------------------------------------------------------------------
    _item(
        "cul_c_01", DOMAIN_CULTURE, "C",
        "Sir, could you tell me the time?",
        he="{formal}{voc}, אני {sp} לדעת מה השעה.",
        he_sp=("סקרן", "סקרנית"),
        he_voc={"male": "אדוני", "female": "גברתי"},
        gu="{voc}, {pd} કૃપા કરીને સમય કહેવો છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        gu_voc={"male": "સાહેબ", "female": "મેડમ"},
        vi="{listen} ơi, {listen} cho {self} biết mấy giờ rồi được không?",
    ),
    _item(
        "cul_c_02", DOMAIN_CULTURE, "C",
        "My friend, do you want to share this meal?",
        he="{formal}{voc}, אני {sp} לשתף איתך את הארוחה.",
        he_sp=("מציע", "מציעה"),
        he_voc={"male": "חברי", "female": "חברתי"},
        gu="{voc}, {pd} આ ભોજન મારી સાથે શેર કરવું છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        gu_voc={"male": "મિત્ર", "female": "મિત્રી"},
        vi="{listen} ơi, {listen} có muốn chia sẻ bữa ăn này với {self} không?",
    ),
    _item(
        "cul_c_03", DOMAIN_CULTURE, "C",
        "Grandma, I am going to the market now.",
        he="{formal}{voc}, אני {sp} לשוק עכשיו.",
        he_sp=("הולך", "הולכת"),
        he_voc={"male": "סבא", "female": "סבתא"},
        gu="{voc}, હું હવે બજારમાં જવા {sp} છું.",
        gu_sp=("નીકળ્યો", "નીકળી"),
        gu_voc={"male": "દાદા", "female": "દાદી"},
        vi="{listen} ơi, {self} đang đi chợ bây giờ.",
    ),
    _item(
        "cul_c_04", DOMAIN_CULTURE, "C",
        "My child, please wear your shoes.",
        he="{formal}{voc}, אני {sp} שתנעל את הנעליים שלך.",
        he_sp=("מבקש", "מבקשת"),
        he_voc={"male": "בני", "female": "בתי"},
        gu="{voc}, {pd} પોતાના પગરખાં પહેરવા જોઈએ; હું {sp} છું.",
        gu_sp=("કહેતો", "કહેતી"),
        gu_voc={"male": "દીકરા", "female": "દીકરી"},
        vi="{listen} ơi, {listen} nhớ mang giày vào.",
    ),
    _item(
        "cul_c_05", DOMAIN_CULTURE, "C",
        "Teacher, I have a question about the lesson.",
        he="{formal}{voc}, אני {sp} לשאול שאלה לגבי השיעור.",
        he_sp=("מבקש", "מבקשת"),
        he_voc={"male": "מורי", "female": "מורתי"},
        gu="{voc}, {pd} પાઠ વિશે એક પ્રશ્ન પૂછવો છે? હું {sp} છું.",
        gu_sp=("પૂછતો", "પૂછતી"),
        gu_voc={"male": "શિક્ષક", "female": "શિક્ષિકા"},
        vi="{listen} ơi, {self} có một câu hỏi về bài học.",
    ),
    _item(
        "cul_c_06", DOMAIN_CULTURE, "C",
        "Neighbor, can you help me carry this?",
        he="{formal}{voc}, אני {sp} שתעזור לי לשאת את זה.",
        he_sp=("מבקש", "מבקשת"),
        he_voc={"male": "שכני", "female": "שכנתי"},
        gu="{voc}, {pd} આ વસ્તુ ઉપાડવામાં મદદ કરવી છે? હું {sp} છું.",
        gu_sp=("વિનંતી કરતો", "વિનંતી કરતી"),
        gu_voc={"male": "પાડોશી", "female": "પાડોશણ"},
        vi="{listen} ơi, {listen} giúp {self} mang vật này được không?",
    ),
]


# ---------------------------------------------------------------------------
# Language-specific mechanics
# ---------------------------------------------------------------------------

GU_LEVELS = {
    "level_1_intimate": {
        "tier": "tu",
        "nom": "તું",
        "erg": "તેં",
        "dat": "તને",
        "poss": "તારું",
    },
    "level_2_polite": {
        "tier": "tame",
        "nom": "તમે",
        "erg": "તમે",
        "dat": "તમને",
        "poss": "તમારું",
    },
    "level_3_honorific": {
        "tier": "aap",
        "nom": "આપ",
        "erg": "આપે",
        "dat": "આપને",
        "poss": "આપનું",
    },
}

GU_OPPOSITE = {
    "level_1_intimate": "level_3_honorific",
    "level_2_polite": "level_1_intimate",
    "level_3_honorific": "level_2_polite",
}

VI_KINSHIP = {
    "anh_em": {"self": "anh", "listen": "em"},
    "chi_em": {"self": "chị", "listen": "em"},
    "con_ong": {"self": "con", "listen": "ông"},
}

VI_C_KINSHIP = {
    ("male", "male"): ("anh_em", "anh", "em"),
    ("male", "female"): ("chu_chau", "chú", "cháu"),
    ("female", "male"): ("chi_em", "chị", "em"),
    ("female", "female"): ("co_chau", "cô", "cháu"),
}

# (northern, southern, kind)
VI_DIALECT_MARKERS = [
    ("nhé", "nha", "particle"),
    ("vâng", "dạ", "prefix"),
    ("bát", "chén", "noun"),
    ("thìa", "muỗng", "noun"),
    ("lợn", "heo", "noun"),
    ("mũ", "nón", "noun"),
    ("ngô", "bắp", "noun"),
    ("cốc", "ly", "noun"),
    ("dứa", "thơm", "noun"),
    ("tất", "vớ", "noun"),
    ("bố", "ba", "noun"),
    ("mẹ", "má", "noun"),
]

VI_NOUN_CLAUSES = {
    "bát": "{self} ăn cơm bằng {d}.",
    "chén": "{self} ăn cơm bằng {d}.",
    "thìa": "{self} dùng {d} để ăn.",
    "muỗng": "{self} dùng {d} để ăn.",
    "lợn": "Nhà {self} nuôi {d}.",
    "heo": "Nhà {self} nuôi {d}.",
    "mũ": "{self} đội {d}.",
    "nón": "{self} đội {d}.",
    "ngô": "{self} mua {d}.",
    "bắp": "{self} mua {d}.",
    "cốc": "{self} uống nước bằng {d}.",
    "ly": "{self} uống nước bằng {d}.",
    "dứa": "{self} ăn {d}.",
    "thơm": "{self} ăn {d}.",
    "tất": "{self} mang {d}.",
    "vớ": "{self} mang {d}.",
    "bố": "{d} của {self} ở nhà.",
    "ba": "{d} của {self} ở nhà.",
    "mẹ": "{d} của {self} nấu ăn.",
    "má": "{d} của {self} nấu ăn.",
}


def _opposite_gender(gender: str) -> str:
    return "female" if gender == "male" else "male"


def _he_formal_prefix(bucket: str) -> str:
    if bucket == "B":
        return "אם יורשה לי, "
    return "בבקשה, "


def _vi_gold(item: dict, index: int, dialect: str, self_word: str, listen_word: str) -> str:
    base = item["vi"].format(self=self_word, listen=listen_word)
    north, south, kind = VI_DIALECT_MARKERS[index % len(VI_DIALECT_MARKERS)]
    marker = north if dialect == "northern" else south
    if kind == "particle":
        return f"{base} {marker}."
    if kind == "prefix":
        return f"{marker}, {base}"
    clause = VI_NOUN_CLAUSES[marker]
    return f"{base} {clause.format(self=self_word, d=marker)}"


def _base_row(item: dict, lang: str, variant_id: str, bucket: str) -> dict:
    row = {key: "" for key in HEADER}
    row.update(
        {
            "item_id": item["id"],
            "variant_id": variant_id,
            "source_en": item["en"],
            "lang": lang,
            "track": item["domain"],
            "bucket": bucket,
            "is_cued_source": "true" if bucket == "C" else "false",
            "relationship": "",
        }
    )
    return row


def hebrew_rows() -> List[dict]:
    rows: List[dict] = []
    for item in ITEMS:
        bucket = item["bucket"]
        for formality in ("default", "formal"):
            for speaker_gender in ("male", "female"):
                listeners = ("male", "female") if bucket == "C" else (None,)
                for listener_gender in listeners:
                    suffix = f"{formality}-{speaker_gender}"
                    if listener_gender is not None:
                        suffix += f"-{listener_gender}"
                    row = _base_row(item, "he", f"{item['id']}-he-{suffix}", bucket)
                    row["formality"] = formality
                    row["speaker_gender"] = speaker_gender
                    row["listener_gender"] = listener_gender or ""
                    prefix = _he_formal_prefix(bucket) if formality == "formal" else ""
                    row["gold_reference"] = item["he"].format(
                        sp=item["he_sp"][speaker_gender],
                        voc=item["he_voc"].get(listener_gender or "", ""),
                        formal=prefix,
                    )
                    if bucket == "C":
                        row["conflict_formality"] = "level_1_intimate"
                        row["conflict_listener_gender"] = _opposite_gender(listener_gender)
                    rows.append(row)
    return rows


def gujarati_rows() -> List[dict]:
    rows: List[dict] = []
    for item in ITEMS:
        bucket = item["bucket"]
        for formality, level in GU_LEVELS.items():
            for speaker_gender in ("male", "female"):
                listeners = ("male", "female") if bucket == "C" else (None,)
                for listener_gender in listeners:
                    suffix = f"{formality}-{speaker_gender}"
                    if listener_gender is not None:
                        suffix += f"-{listener_gender}"
                    row = _base_row(item, "gu", f"{item['id']}-gu-{suffix}", bucket)
                    row["formality"] = formality
                    row["speaker_gender"] = speaker_gender
                    row["listener_gender"] = listener_gender or ""
                    row["pronoun_tier"] = level["tier"]
                    row["gold_reference"] = item["gu"].format(
                        sp=item["gu_sp"][speaker_gender],
                        p=level["nom"],
                        pd=level["dat"],
                        pe=level["erg"],
                        poss=level["poss"],
                        voc=item["gu_voc"].get(listener_gender or "", ""),
                    )
                    if bucket == "C":
                        row["conflict_formality"] = GU_OPPOSITE[formality]
                        row["conflict_speaker_gender"] = _opposite_gender(speaker_gender)
                    rows.append(row)
    return rows


def vietnamese_rows() -> List[dict]:
    rows: List[dict] = []
    for index, item in enumerate(ITEMS):
        bucket = item["bucket"]
        for dialect in ("northern", "southern"):
            if bucket == "C":
                for speaker_gender in ("male", "female"):
                    for listener_gender in ("male", "female"):
                        pair, self_word, listen_word = VI_C_KINSHIP[
                            (speaker_gender, listener_gender)
                        ]
                        variant_id = (
                            f"{item['id']}-vi-{dialect}-{speaker_gender}-{listener_gender}"
                        )
                        row = _base_row(item, "vi", variant_id, bucket)
                        row["formality"] = "default"
                        row["speaker_gender"] = speaker_gender
                        row["listener_gender"] = listener_gender
                        row["dialect"] = dialect
                        row["kinship_pair"] = pair
                        row["gold_reference"] = _vi_gold(
                            item, index, dialect, self_word, listen_word
                        )
                        row["conflict_dialect"] = (
                            "southern" if dialect == "northern" else "northern"
                        )
                        row["conflict_listener_gender"] = _opposite_gender(listener_gender)
                        rows.append(row)
            else:
                for pair, words in VI_KINSHIP.items():
                    variant_id = f"{item['id']}-vi-{dialect}-{pair}"
                    row = _base_row(item, "vi", variant_id, bucket)
                    row["formality"] = "default"
                    row["dialect"] = dialect
                    row["kinship_pair"] = pair
                    row["gold_reference"] = _vi_gold(
                        item, index, dialect, words["self"], words["listen"]
                    )
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Validation + writing
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_items() -> List[str]:
    problems: List[str] = []
    if len(ITEMS) != 36:
        problems.append(f"expected 36 base items, got {len(ITEMS)}")
    if len({item["id"] for item in ITEMS}) != 36:
        problems.append("item IDs are not unique")
    for domain in (DOMAIN_FAMILY, DOMAIN_CULTURE):
        domain_items = [item for item in ITEMS if item["domain"] == domain]
        if len(domain_items) != 18:
            problems.append(f"{domain}: expected 18 items, got {len(domain_items)}")
        for bucket in ("A", "B", "C"):
            count = sum(1 for item in domain_items if item["bucket"] == bucket)
            if count != 6:
                problems.append(f"{domain}/{bucket}: expected 6 items, got {count}")
    cued = [item for item in ITEMS if item["bucket"] == "C"]
    ambiguous = [item for item in ITEMS if item["bucket"] in ("A", "B")]
    if len(cued) != 12 or len(ambiguous) != 24:
        problems.append(f"expected 24 ambiguous + 12 cued, got {len(ambiguous)}+{len(cued)}")
    return problems


def _validate_rows(label: str, rows: List[dict], expected: int) -> List[str]:
    problems: List[str] = []
    if len(rows) != expected:
        problems.append(f"{label}: expected {expected} rows, got {len(rows)}")
    if len({row["variant_id"] for row in rows}) != len(rows):
        problems.append(f"{label}: duplicate variant_id values")
    if any(list(row.keys()) != HEADER for row in rows):
        problems.append(f"{label}: row keys/order differ from header")

    required_common = ("item_id", "variant_id", "source_en", "lang", "track", "bucket",
                       "is_cued_source", "gold_reference")
    for column in required_common:
        if any(not str(row[column]).strip() for row in rows):
            problems.append(f"{label}: empty required field {column}")

    for row in rows:
        if row["bucket"] == "C":
            if row["is_cued_source"] != "true":
                problems.append(f"{label}/{row['variant_id']}: C row is not cued")
                break
        else:
            if row["is_cued_source"] != "false":
                problems.append(f"{label}/{row['variant_id']}: non-C row is cued")
                break
        if row["bucket"] != "C":
            conflict_fields = ("conflict_formality", "conflict_speaker_gender",
                               "conflict_listener_gender", "conflict_dialect")
            if any(row[field] for field in conflict_fields):
                problems.append(f"{label}/{row['variant_id']}: conflict on non-C row")
                break

    for row in rows:
        if row["lang"] == "he":
            if not row["formality"] or not row["speaker_gender"]:
                problems.append(f"{label}: empty he dimension on {row['variant_id']}")
                break
            if row["bucket"] == "C" and not row["listener_gender"]:
                problems.append(f"{label}: empty he C listener on {row['variant_id']}")
                break
            if row["bucket"] == "C" and (
                row["conflict_formality"] != "level_1_intimate"
                or not row["conflict_listener_gender"]
            ):
                problems.append(f"{label}: missing he conflict on {row['variant_id']}")
                break
        elif row["lang"] == "gu":
            if not row["formality"] or not row["speaker_gender"] or not row["pronoun_tier"]:
                problems.append(f"{label}: empty gu dimension on {row['variant_id']}")
                break
            if row["bucket"] == "C" and not row["listener_gender"]:
                problems.append(f"{label}: empty gu C listener on {row['variant_id']}")
                break
            if row["bucket"] == "C" and (
                not row["conflict_formality"] or not row["conflict_speaker_gender"]
            ):
                problems.append(f"{label}: missing gu conflict on {row['variant_id']}")
                break
        elif row["lang"] == "vi":
            if not row["dialect"] or not row["kinship_pair"]:
                problems.append(f"{label}: empty vi dimension on {row['variant_id']}")
                break
            if row["bucket"] == "C" and (
                not row["speaker_gender"] or not row["listener_gender"]
            ):
                problems.append(f"{label}: empty vi C gender on {row['variant_id']}")
                break
            if row["bucket"] == "C" and (
                not row["conflict_dialect"] or not row["conflict_listener_gender"]
            ):
                problems.append(f"{label}: missing vi conflict on {row['variant_id']}")
                break
    return problems


def _validate_vietnamese_dialects(rows: List[dict]) -> List[str]:
    """Every northern/southern sibling must differ on a known marker."""
    problems: List[str] = []
    all_markers = {word for pair in VI_DIALECT_MARKERS for word in pair[:2]}
    by_key: Dict[tuple, Dict[str, str]] = {}
    for row in rows:
        if row["lang"] != "vi":
            continue
        key = (
            row["item_id"],
            row["bucket"],
            row["kinship_pair"],
            row["speaker_gender"],
            row["listener_gender"],
        )
        by_key.setdefault(key, {})[row["dialect"]] = row["gold_reference"]

    for key, pair in by_key.items():
        if set(pair) != {"northern", "southern"}:
            problems.append(f"vi: missing dialect sibling for {key}")
            continue
        north, south = pair["northern"], pair["southern"]
        if north == south:
            problems.append(f"vi: identical dialect gold for {key}")
            continue
        if not any(marker in north or marker in south for marker in all_markers):
            problems.append(f"vi: no listed dialect marker for {key}")
    return problems


def main() -> int:
    problems: List[str] = []
    problems.extend(_validate_items())

    sheets = {
        "he": (OUT_DIR / "he_curation.csv", hebrew_rows(), 192),
        "gu": (OUT_DIR / "gu_curation.csv", gujarati_rows(), 288),
        "vi": (OUT_DIR / "vi_curation.csv", vietnamese_rows(), 240),
    }

    for label, (path, rows, expected) in sheets.items():
        problems.extend(_validate_rows(label, rows, expected))
        if label == "vi":
            problems.extend(_validate_vietnamese_dialects(rows))
        _write_csv(path, rows)

    if problems:
        print("VALIDATION FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("VALIDATION PASSED")
    print(f"repo root: {REPO_ROOT}")
    for label, (path, rows, expected) in sheets.items():
        cued_items = len({row["item_id"] for row in rows if row["is_cued_source"] == "true"})
        cued_rows = sum(1 for row in rows if row["is_cued_source"] == "true")
        print(
            f"{path.relative_to(REPO_ROOT)}: {len(rows)} data rows "
            f"(expected {expected}); cued base items={cued_items}; "
            f"cued rows={cued_rows}; no empty required fields"
        )
    print("base items: 36 (24 ambiguous + 12 cued; 18 family + 18 daily life)")
    print("variant_id globally unique; dialect siblings differ on listed markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
