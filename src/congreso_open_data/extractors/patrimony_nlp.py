from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

Classification = dict[str, str]
CategoryResolver = Callable[[str], str]

_PARTY_RE = re.compile(r"\b(psoe|psrm|pce|pp|vox|sumar|podemos|junts|erc)\b")
_AMOUNT_RE = re.compile(
    r"(?<!\d)(\d+(?:[.\s]\d{3})*,\d{2,4})\s*(?:\u20ac|eur(?:os?)?)?",
    re.I,
)
_PERCENT_RE = re.compile(r"\b(\d{1,3}(?:,\d{1,4})?)\s*%")


@dataclass(frozen=True)
class PatrimonyRule:
    family: str
    category: str | CategoryResolver
    tokens: tuple[str, ...]
    document_kinds: tuple[str, ...] = ()
    role_resolver: Callable[[str], str | None] | None = None
    subcategory_resolver: Callable[[str], str | None] | None = None

    def applies_to(self, document_kind: str) -> bool:
        return not self.document_kinds or document_kind in self.document_kinds

    def matches(self, normalized_line: str, document_kind: str) -> bool:
        return self.applies_to(document_kind) and any(
            token in normalized_line for token in self.tokens
        )

    def classify(self, line: str, normalized_line: str) -> Classification:
        category = self.category(normalized_line) if callable(self.category) else self.category
        result: Classification = {
            "item_family": self.family,
            "item_category": category,
            "label": label_from_line(line),
        }
        if self.subcategory_resolver:
            subcategory = self.subcategory_resolver(normalized_line)
            if subcategory:
                result["item_subcategory"] = subcategory
        if self.role_resolver:
            role = self.role_resolver(normalized_line)
            if role:
                result["role"] = role
        return result


def classify_patrimony_line(line: str, document_kind: str) -> Classification | None:
    normalized = fold_text(line)
    for rule in _RULES:
        if rule.matches(normalized, document_kind):
            if rule.family == "company_interest" and _is_public_party_or_academic_council(
                normalized
            ):
                continue
            if (
                rule.family == "income"
                and document_kind
                in {
                    "activities",
                    "economic_interests",
                }
                and is_negative_income_fragment(normalized)
            ):
                return None
            return rule.classify(line, normalized)
    return None


def is_patrimony_boilerplate(line: str) -> bool:
    normalized = fold_text(line)
    compact = re.sub(r"\s+", "", normalized)
    loose = boilerplate_fingerprint(line)
    if re.fullmatch(r"[a-z]\)\s+.+", normalized) and not _has_amount(line):
        return True
    if re.fullmatch(r"\d+\.\s*declaracion de actividades,?\s+de\s+.+", normalized):
        return True
    if normalized.strip(" .,:;-") == "registro de intereses - actividades":
        return True
    if normalized.strip(" .,:;-") in _SHORT_BOILERPLATE_LINES:
        return True
    if any(token in normalized for token in _BOILERPLATE_SUBSTRINGS):
        return True
    if any(token in compact or token in loose for token in _COMPACT_BOILERPLATE):
        return True
    if "saldo" in loose and "deposit" in loose and "cuenta" in loose:
        return True
    if "indicarlaclasededeposito" in loose:
        return True
    if (
        "deposito" in loose
        and ("pagars" in loose or "pagares" in loose)
        and "demas" in loose
        and not _has_amount(line)
    ):
        return True
    if "saldomedio" in loose and "cuentascorrientes" in loose:
        return True
    if "indicar" in loose and "matricula" in loose and "vehiculo" in loose:
        return True
    if "indicar" in loose and "piso" in loose and "provincia" in loose:
        return True
    if "piso" in loose and "plaza" in loose and ("provincia" in loose or "pais" in loose):
        return True
    if "deudapublica" in loose and "obligaciones" in loose:
        return True
    if loose.startswith("deudap") and "obligaciones" in loose:
        return True
    if loose.startswith("deudasy") and (
        "patrimon" in loose or "paibim" in loose or "patn" in loose
    ):
        return True
    if (
        "deudas" in loose
        and ("obligac" in loose or "oblgac" in loose)
        and ("contratos" in loose or "oontratos" in loose)
    ):
        return True
    if loose in _LOOSE_BOILERPLATE_LINES:
        return True
    return normalized.strip(" .,:;").lower() in _NORMALIZED_BOILERPLATE_LINES


def has_patrimony_evidence_signal(
    line: str,
    classification: Classification,
    *,
    document_kind: str,
) -> bool:
    normalized = fold_text(line)
    if document_kind in {"activities", "economic_interests"}:
        return True
    if _has_amount(line) or _PERCENT_RE.search(line):
        return True
    if classification["item_family"] in {"position", "company_interest"}:
        return True
    if classification["item_category"] in {"real_estate", "vehicle"}:
        return _has_location_or_acquisition_signal(normalized) or len(line) >= 24
    return _has_declared_asset_signal(normalized)


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def label_from_line(line: str) -> str:
    label = re.split(r"\s{2,}|\t", line.strip(), maxsplit=1)[0]
    label = re.sub(r"\s+", " ", label)
    return label[:120]


def boilerplate_fingerprint(line: str) -> str:
    loose = fold_text(line)
    loose = re.sub(r"[^a-z0-9]+", "", loose)
    replacements = {
        "lndicar": "indicar",
        "galdo": "saldo",
        "galdq": "saldo",
        "saho": "saldo",
        "debeper": "debeser",
        "depeser": "debeser",
        "sumatoriq": "sumatorio",
        "sumalorio": "sumatorio",
        "todoq": "todos",
        "toqos": "todos",
        "todoo": "todos",
        "dep6sitos": "depositos",
        "dep0sitos": "depositos",
        "depoqitos": "depositos",
        "depqsitos": "depositos",
        "dplitos": "depositos",
        "cgentas": "cuentas",
        "cuentascarrientes": "cuentascorrientes",
        "cqrrientes": "corrientes",
        "lntereses": "intereses",
        "meqio": "medio",
        "sado": "saldo",
        "lemar": "tomar",
        "temar": "tomar",
        "refele": "refere",
        "p0blica": "publica",
        "p6blica": "publica",
        "priblica": "publica",
        "prfblica": "publica",
        "prlblica": "publica",
        "pfblica": "publica",
        "pdblica": "publica",
        "pblica": "publica",
        "bllca": "blica",
        "obligacionee": "obligaciones",
        "obligacionas": "obligaciones",
        "obligaclones": "obligaciones",
        "obllgaclones": "obligaciones",
        "obitgackies": "obligaciones",
        "obligacaones": "obligaciones",
        "deravadas": "derivadas",
        "vivlenda": "vivienda",
        "matr1cula": "matricula",
        "matricuia": "matricula",
        "matricu1a": "matricula",
        "incluirvehiculosr": "incluirvehiculos",
        "incluirvehiculos": "incluirvehiculos",
        "vehiculossr": "vehiculos",
        "embarcacioneso": "embarcaciones",
    }
    for old, new in replacements.items():
        loose = loose.replace(old, new)
    return loose


def _has_amount(line: str) -> bool:
    return bool(_AMOUNT_RE.search(line))


def _has_location_or_acquisition_signal(normalized: str) -> bool:
    return any(
        token in normalized
        for token in (
            "provincia",
            "municipio",
            "localidad",
            "pais",
            "madrid",
            "barcelona",
            "adquisicion",
            "herencia",
            "donacion",
            "compraventa",
            "ganancial",
            "pleno dominio",
            "usufructo",
        )
    )


def _has_declared_asset_signal(normalized: str) -> bool:
    return any(
        token in normalized
        for token in (
            "saldo",
            "valor",
            "titular",
            "participacion",
            "porcentaje",
            "acciones de",
            "cuenta en",
            "deposito en",
            "plan de pensiones",
            "seguro de vida",
        )
    )


def _income_category(normalized: str) -> str:
    if any(
        token in normalized for token in ("medios de comunicacion", "tertuliano", "radiofonico")
    ):
        return "private_activity"
    if any(
        token in normalized
        for token in ("congreso", "senado", "ayuntamiento", "ministerio", "public")
    ):
        return "public_salary"
    if any(token in normalized for token in ("alquiler", "arrendamiento")):
        return "rental_income"
    if "dividendo" in normalized:
        return "dividends"
    if "interes" in normalized:
        return "interest"
    if "pension" in normalized:
        return "pension"
    if "docencia" in normalized or "universidad" in normalized:
        return "teaching"
    if "conferencia" in normalized:
        return "speaking"
    if "derechos de autor" in normalized:
        return "royalties"
    if "actividad privada" in normalized:
        return "private_activity"
    return "other_income"


def _asset_category(normalized: str) -> str:
    if "cuenta" in normalized:
        return "bank_account"
    if "deposito" in normalized:
        return "deposit"
    if "acciones" in normalized or "participaciones" in normalized or "valores" in normalized:
        return "shares"
    if "fondo" in normalized:
        return "fund"
    if "plan de pensiones" in normalized:
        return "pension_plan"
    if "seguro" in normalized:
        return "insurance"
    if "cripto" in normalized:
        return "crypto"
    return "other_asset"


def _liability_category(normalized: str) -> str:
    if "hipoteca" in normalized:
        return "mortgage"
    if "prestamo" in normalized:
        return "loan"
    if "credito" in normalized:
        return "credit"
    if "aval" in normalized:
        return "guarantee"
    return "other_liability"


def _position_category(normalized: str) -> str:
    if any(
        token in normalized
        for token in (
            "medios de comunicacion",
            "comunicacion social",
            "radiofonico",
            "televisivo",
            "tertulia",
            "editorial",
        )
    ):
        return "media"
    if any(
        token in normalized
        for token in (
            "ayuntamiento",
            "parlamento",
            "asamblea",
            "diputacion",
            "congreso",
            "senado",
            "ministerio",
            "cabildo",
            "concejal",
            "femp",
            "comite europeo de las regiones",
            "comision de patrimonio",
            "comarcal",
        )
    ):
        return "public_office"
    if any(token in normalized for token in ("partido", "comision ejecutiva", "juventudes")):
        return "party"
    if _PARTY_RE.search(normalized):
        return "party"
    if "fundacion" in normalized:
        return "foundation"
    if any(
        token in normalized
        for token in ("universidad", "docente", "docencia", "academica", "profesorado", "tesis")
    ):
        return "university"
    if "consejo" in normalized:
        return "board"
    if "asesor" in normalized:
        return "advisory"
    if any(
        token in normalized
        for token in (
            "empresa",
            "sociedad",
            "autonom",
            "consultor",
            "actividad profesional",
        )
    ):
        return "private_company"
    if any(
        token in normalized
        for token in (
            "mutualidad",
            "colegio oficial",
            "comite cientifico",
            "junta directiva",
        )
    ):
        return "association"
    return "other_position"


def _company_interest_category(normalized: str) -> str:
    if any(token in normalized for token in ("participacion", "acciones", "socio")):
        return "shareholding"
    if any(token in normalized for token in ("administrador", "directivo", "gerencia", "gestora")):
        return "director"
    if any(token in normalized for token in ("consejo", "consejera")):
        return "board_member"
    if any(token in normalized for token in ("asesor", "consultor")):
        return "advisor"
    if any(
        token in normalized
        for token in ("empleado", "trabajador", "excedencia", "tecnica", "director de negocio")
    ):
        return "employee"
    return "other_company_interest"


def _is_public_party_or_academic_council(normalized: str) -> bool:
    if "consejo" not in normalized:
        return False
    if any(
        token in normalized
        for token in (
            "empresa",
            "sociedad",
            "mercantil",
            "s.a",
            "s.l",
            " sa",
            " sl",
            "compania",
        )
    ):
        return False
    return bool(
        _PARTY_RE.search(normalized)
        or any(
            token in normalized
            for token in (
                "alcaldes",
                "ayuntamiento",
                "cabildo",
                "ciudad",
                "organismo autonomo",
                "sector publico",
                "universidad",
                "consejo social",
            )
        )
    )


def _real_estate_subcategory(normalized: str) -> str | None:
    for token in ("vivienda", "piso", "local", "garaje", "finca", "rustica", "urbana"):
        if token in normalized:
            return token
    return None


def _role_from_line(normalized: str) -> str | None:
    for role in ("administrador", "consejero", "asesor", "profesor", "alcalde", "concejal"):
        if role in normalized:
            return role
    return None


def is_negative_income_fragment(normalized: str) -> bool:
    return any(token in normalized for token in _NEGATIVE_INCOME_TOKENS)


_RULES = (
    PatrimonyRule(
        family="liability",
        category=_liability_category,
        tokens=("hipoteca", "prestamo", "credito", "deuda", "aval"),
        document_kinds=("assets_income",),
    ),
    PatrimonyRule(
        family="asset",
        category="real_estate",
        tokens=("inmueble", "vivienda", "piso", "local", "garaje", "finca", "rustica", "urbana"),
        document_kinds=("assets_income",),
        subcategory_resolver=_real_estate_subcategory,
    ),
    PatrimonyRule(
        family="asset",
        category="vehicle",
        tokens=("vehiculo", "turismo", "automovil", "coche", "motocicleta", "embarcacion"),
        document_kinds=("assets_income",),
    ),
    PatrimonyRule(
        family="asset",
        category=_asset_category,
        tokens=(
            "acciones",
            "participaciones",
            "valores",
            "fondo",
            "deposito",
            "cuenta",
            "plan de pensiones",
            "seguro",
            "cripto",
        ),
        document_kinds=("assets_income", "economic_interests"),
    ),
    PatrimonyRule(
        family="company_interest",
        category=_company_interest_category,
        tokens=("empresa", "sociedad", "consejo"),
        document_kinds=("activities", "economic_interests"),
        role_resolver=_role_from_line,
    ),
    PatrimonyRule(
        family="position",
        category=_position_category,
        tokens=(
            "cargo",
            "actividad",
            "fundacion",
            "universidad",
            "asesor",
            "partido",
            "vicepresidente",
            "presidente",
            "miembro",
            "tesorero",
            "vicesecretaria",
            "secretario",
            "vocal",
            "gerencia",
            "consejera",
            "comision ejecutiva",
            "medios de comunicacion",
            "comunicacion social",
            "tertuliano",
            "tertulia",
            "radiofonico",
            "televisivo",
            "colaboracion",
            "colaboraciones",
            "obrero espanol",
            "cabildo",
            "comarcal",
            "colegio oficial",
            "junta directiva",
        ),
        document_kinds=("activities", "economic_interests"),
        role_resolver=_role_from_line,
    ),
    PatrimonyRule(
        family="income",
        category=_income_category,
        tokens=(
            "alquiler",
            "arrendamiento",
            "dividendo",
            "intereses",
            "pension",
            "retribucion",
            "salario",
            "rendimiento",
            "docencia",
            "conferencia",
            "derechos de autor",
        ),
    ),
)

_NEGATIVE_INCOME_TOKENS = (
    "sin retribucion",
    "sin remuneracion",
    "no recibe ninguna retribucion",
    "percepcion de ninguna retribucion",
    "renunciando a la retribucion",
    "renunciando al salario",
    "renuncia al salario",
    "renunciando a las retribuciones",
    "sin percibir retribucion",
    "sin percibir salario",
    "sin haber percibido ninguna retribucion",
    "sin percibir ningun tipo de retribucion",
    "sin relacion laboral o retribucion",
    "no remunerado",
    "con o sin retribucion",
)

_SHORT_BOILERPLATE_LINES = {
    "retribucion",
    "retribucion alguna",
    "sin retribucion",
    "sin remuneracion",
}

_BOILERPLATE_SUBSTRINGS = (
    "articulo 159.2",
    "actividades prohibidas",
    "condicion de parlamentario",
    "no menoscabar la dedicacion parlamentaria",
    "la autorizacion de esta actividad",
    "las actividades privadas se autorizan",
    "indicar si es piso",
    "indicar provincia donde",
    "para bienes radicados en",
    "pleno dominio",
    "nuda propiedad",
    "usufructo",
    "el saldo debe ser",
    "se puede tomar como referencia",
    "debe aplicarse a todas las cuentas",
    "no indicar matricula",
    "valor a declarar debe ser",
    "debe reflejarse el valor de cotizacion",
    "debe indicarse la fecha elegida",
    "si no hubiese balance anual",
    "otras deudas y obligaciones derivadas",
    "formulario rellenado con datos ficticios",
    "rentas que han de declararse",
    "se excluiran las percepciones",
    "deben incluirse, en su caso, las percepciones",
    "retribuciones, cualquiera que",
    "intereses o rendimientos de",
    "deudas y obligaciones patrimoniales",
    "deuda publica, obligaciones",
)

_COMPACT_BOILERPLATE = (
    "deudasyobligacionespatrimoniales",
    "deudasyobligaclonespatrimoniales",
    "deudasyobligacionespatrimoniale",
    "otrasdeudasyobligacionesderivadas",
    "retribucionesdinerarias",
    "indicarsiespiso",
    "indicarmatricula",
    "incluirvehiculos",
    "saldodebeser",
    "sumatoriodetodoslosdepositos",
    "saldomediodelascuentascorrientes",
    "deudapublicaobligaciones",
    "bienespatrimonialesdelparlamentario",
    "depositosencuentascorrientesodeahorro",
    "cuentasfinancierasyotrostiposdeimposiciones",
    "otrostiposdeimposiciones",
    "depositose",
    "otrosbienesoderechos",
    "clasedebienodescripciondelbienoderecho",
    "vehiculosembarcacionesyaeronaves",
    "valoresequivalentes",
    "accionesyparticipaciones",
    "entidadyelvalordelasaccionesoparticipaciones",
    "dividendosyparticipacionen",
    "prestamosdescripcionyacreedor",
    "registrodeintereses",
    "rentaspercibidasporelparlamentario",
    "debenincluirseensucasolaspercepciones",
    "percepcionescobradasporplanesdepensiones",
    "interesesorendimientosde",
    "retribucionescualquieraque",
    "cuentasdepositosyactivos",
)

_LOOSE_BOILERPLATE_LINES = {
    "inmuebles",
    "bienesinmuebles",
    "rustica",
    "urbana",
    "acciones",
    "accioneso",
    "participaciones",
}

_NORMALIZED_BOILERPLATE_LINES = {
    "actividades",
    "cargos publicos",
    "deuda publica obligaciones",
    "deuda publica",
    "inmuebles",
    "bienes inmuebles",
    "rustica",
    "urbana",
    "acciones",
    "participaciones",
}
