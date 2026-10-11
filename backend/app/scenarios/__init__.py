"""场景定义模块"""

from app.scenarios.base import BaseScenario, ScenarioRegistry
from app.scenarios.radiology import RadiologyScenario
from app.scenarios.medication import MedicationScenario
from app.scenarios.news import NewsScenario
from app.scenarios.finance import FinanceScenario
from app.scenarios.medical import MedicalScenario
from app.scenarios.sales import SalesScenario
from app.scenarios.customer_service import CustomerServiceScenario

# 注册场景
ScenarioRegistry.register("radiology", RadiologyScenario)
ScenarioRegistry.register("medication", MedicationScenario)
ScenarioRegistry.register("news", NewsScenario)
ScenarioRegistry.register("finance", FinanceScenario)
ScenarioRegistry.register("medical", MedicalScenario)
ScenarioRegistry.register("sales", SalesScenario)
ScenarioRegistry.register("customer_service", CustomerServiceScenario)

__all__ = [
    "BaseScenario",
    "ScenarioRegistry",
    "RadiologyScenario",
    "MedicationScenario",
    "NewsScenario",
    "FinanceScenario",
    "MedicalScenario",
    "SalesScenario",
    "CustomerServiceScenario",
]
