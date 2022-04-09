# Copyright (C) 2022 Matthew Marting
# SPDX-License-Identifier: GPL-3.0-or-later

from argparse import ArgumentParser
from collections import defaultdict
from collections import deque
from collections.abc import Iterable
from configparser import ConfigParser
from configparser import SectionProxy
from contextlib import AbstractContextManager
from contextlib import contextmanager
import csv
from decimal import Decimal as d
from decimal import ROUND_UP
from decimal import localcontext
from functools import wraps
from itertools import chain
from itertools import count
from itertools import repeat
import json
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Generic
from typing import NamedTuple
from typing import Optional
from typing import Type
from typing import TypeVar
import lxml.html
from mypy_extensions import DefaultNamedArg
from numpy import array
from numpy import ndarray
import requests
from requests import Response
from requests import Session
from requests.adapters import HTTPAdapter
from requests.adapters import Retry
import requests.api
from selenium.common.exceptions import WebDriverException
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager


ContextManager = TypeVar("ContextManager", bound=AbstractContextManager)
Return = TypeVar("Return")


class ContextManagerWrapper(AbstractContextManager, Generic[ContextManager]):
    thing: ContextManager

    def __init__(self, thing: ContextManager, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thing = thing

    def __enter__(self) -> None:
        self.value = self.thing.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        try:
            del self.value
        except AttributeError:
            pass
        return self.thing.__exit__(exc_type, exc_value, traceback)

    def new(self, thing: ContextManager) -> None:
        self.__exit__(None, None, None)
        self.thing = thing
        self.__enter__()


class Retries(NamedTuple):
    total: int
    backoff_factor: float
    backoff_max: float

    def _get_adapter(self) -> HTTPAdapter:
        retries = Retry(total=self.total, backoff_factor=self.backoff_factor)
        retries.DEFAULT_BACKOFF_MAX = self.backoff_max
        return HTTPAdapter(max_retries=retries)

    def request(self, method: str, url: str, **kwargs) -> Response:
        with Session() as session:
            session.mount("https://", self._get_adapter())
            session.mount("http://", self._get_adapter())
            return session.request(method, url, **kwargs)


def request(method: str, url: str, section: SectionProxy, **kwargs) -> Response:
    return Retries(
        #
        total=int(section["request.retries.total"]),
        backoff_factor=float(section["request.retries.backoff_factor"]),
        backoff_max=float(section["request.retries.backoff_max"]),
    ).request(
        method,
        url,
        #
        timeout=float(section["request.timeout"]),
        **kwargs,
    )


@contextmanager
def requestcontext(section: SectionProxy) -> Generator[None, None, None]:
    request_ = requests.api.request
    requests.request = requests.api.request = lambda method, url, **kwargs: request(method, url, section, **kwargs)
    yield
    requests.request = requests.api.request = request_


def get_driver(config: ConfigParser, headless: Optional[bool] = None) -> Firefox:
    if headless is None:
        headless = False
    options = Options()
    options.headless = headless
    with requestcontext(config["request"]):
        requests.request = lambda method, url, **kwargs: request(method, url, config["request"], **kwargs)
        return Firefox(options=options, service=Service(GeckoDriverManager().install()))


def get_addons_directory() -> TemporaryDirectory:
    return TemporaryDirectory()


class WebDriver:
    _config: ConfigParser
    _driver: Firefox

    def _addon_url(self, addon: str) -> str:
        webpage_url = f"https://addons.mozilla.org/en-US/firefox/addon/{addon}"
        webpage_response = request(
            "get",
            webpage_url,
            #
            self._config["firefox.addon_webpage_request"],
        )
        webpage_response.raise_for_status()
        webpage_html = lxml.html.fromstring(webpage_response.text)
        download_file = webpage_html.xpath(
            #
            "//a[contains(descendant-or-self::*, 'Download file')]"
        )[0]
        return download_file.attrib["href"]

    def __init__(
        self,
        config: ConfigParser,
        driver: Firefox,
        addons_directory: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._config = config
        self._driver = driver
        for key, addon in (
            #
            self._config.items("firefox.addons")
        ):
            if not key.startswith("addon"):
                continue
            url = self._addon_url(addon)
            response = request(
                "get",
                url,
                #
                self._config["firefox.addon_request"],
            )
            response.raise_for_status()
            basename = os.path.basename(url)
            path = os.path.join(addons_directory, basename)
            with open(path, "wb") as file:
                file.write(response.content)
            self._driver.install_addon(path, temporary=True)


class MorningstarWebDriver(WebDriver):
    _FIELD_COUNT: int = 10
    _l: list[Callable[["MorningstarWebDriver"], Any]] = []
    _instant_x_ray: bool
    _cache: dict[tuple[tuple[str, d], ...], dict[str, Any]]

    def __init__(
        self,
        credentials: ConfigParser,
        cache: dict[tuple[tuple[str, d], ...], dict[str, Any]],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cache = cache

        self._driver.get(
            #
            "https://www.morningstar.com/auth-init?"
            "destination=https://www.morningstar.com/instant-x-ray"
        )
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.email_input_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//input[@id='emailInput']")
            )
        ).send_keys(
            #
            credentials["morningstar.credentials"]["email"]
        )
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.password_input_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//input[@id='passwordInput']")
            )
        ).send_keys(
            #
            credentials["morningstar.credentials"]["password"]
        )
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.sign_in_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//button[contains(descendant-or-self::*, 'Sign In')]")
            )
        ).click()

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.instant_x_ray_iframe_wait"]["wait.timeout"]),
        ).until(
            EC.frame_to_be_available_and_switch_to_it(
                #
                (By.ID, "instant-x-ray")
            )
        )

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.holding_value_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Holding Value')]")
            )
        ).click()
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.percentage_value_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Percentage Value %')]")
            )
        ).click()

        self._instant_x_ray = False

    def _show_instant_x_ray(self, portfolio: dict[str, d]) -> None:
        assert not self._instant_x_ray

        reset, t0 = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.reset_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.all_of(
                EC.element_to_be_clickable(
                    #
                    (By.XPATH, "//a[contains(descendant-or-self::*, 'Reset')]")
                ),
                EC.element_to_be_clickable(
                    #
                    (By.XPATH, f"//input[@id='t0']")
                ),
            )
        )
        reset.send_keys(Keys.ENTER)
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.t0_stale_wait"]["wait.timeout"]),
        ).until(
            EC.staleness_of(
                #
                t0
            )
        )

        entry_fields = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.entry_fields_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.all_of(
                *(
                    EC.element_to_be_clickable(
                        #
                        (By.XPATH, f"//input[@id='t{i}']")
                    )
                    for i in range(0, 0 + self._FIELD_COUNT)
                ),
                *(
                    EC.element_to_be_clickable(
                        #
                        (By.XPATH, f"//input[@id='p{i}']")
                    )
                    for i in range(0, 0 + self._FIELD_COUNT)
                ),
            )
        )
        entry_fields = deque(entry_fields)
        t = tuple(entry_fields.popleft() for _ in range(self._FIELD_COUNT))
        p = tuple(entry_fields.popleft() for _ in range(self._FIELD_COUNT))
        for i, (portfolio_ticker_symbol, weight) in enumerate(portfolio.items()):
            t[i].send_keys(portfolio_ticker_symbol)
            p[i].send_keys(str(100 * weight))
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.show_instant_x_ray_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Show Instant X-Ray')]")
            )
        ).send_keys(
            Keys.ENTER
        )

        self._instant_x_ray = True

    def _edit_holdings(self) -> None:
        assert self._instant_x_ray

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.edit_holdings_clickable_wait"]["wait.timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Edit Holdings')]")
            )
        ).send_keys(
            Keys.ENTER
        )

        self._instant_x_ray = False

    @staticmethod
    def _withportfolio(
        l: list[Callable[["MorningstarWebDriver"], Any]]
    ) -> Callable[
        [Callable[["MorningstarWebDriver"], Return]],
        #
        Callable[["MorningstarWebDriver", dict[str, d]], Return],
    ]:
        def decorator(
            f: Callable[["MorningstarWebDriver"], Return]
        ) -> Callable[["MorningstarWebDriver", dict[str, d]], Return]:
            l.append(f)

            @wraps(f)
            def wrapper(self, portfolio: dict[str, d]) -> Return:
                hashable_portfolio = tuple(sorted(portfolio.items()))
                try:
                    portfolio_cache = self._cache[hashable_portfolio]
                except KeyError:
                    if self._instant_x_ray:
                        self._edit_holdings()
                    self._show_instant_x_ray(portfolio)
                    portfolio_cache = self._cache[hashable_portfolio] = {}
                    for g in self._l:
                        portfolio_cache[g.__name__] = g(self)
                return portfolio_cache[f.__name__]

            return wrapper

        return decorator

    @_withportfolio(_l)
    def asset_allocation(self) -> tuple[int, ...]:
        assert self._instant_x_ray

        asset_allocation = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.asset_allocation_visible_wait"]["wait.timeout"]),
        ).until(
            EC.visibility_of_element_located(
                #
                (By.XPATH, "//table[@class='assetTbl']")
            )
        )
        return tuple(
            map(
                lambda element: int(element.text),
                asset_allocation.find_elements(
                    #
                    *(By.XPATH, ".//tr[@class='TextData']//td[@class='long']")
                ),
            )
        )

    @_withportfolio(_l)
    def style_box(self) -> tuple[int, ...]:
        assert self._instant_x_ray

        style_box = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar_firefox.style_box_visible_wait"]["wait.timeout"]),
        ).until(
            EC.visibility_of_element_located(
                #
                (By.XPATH, "//div[@class='pmx_ssdivers_cola']//div[@class='pmx_boxw']")
            )
        )
        return tuple(
            map(
                lambda element: int(element.text),
                style_box.find_elements(
                    #
                    *(By.XPATH, ".//div[@class='pmx_box' or @class='pmx_boxb']")
                ),
            )
        )


def dummy_portfolio(
    ticker_symbol: str,
    control_ticker_symbol: Optional[str] = None,
    control_weight: Optional[d] = None,
) -> dict[str, d]:
    if control_weight is None:
        control_weight = d("0")
    if not 0 <= control_weight <= 1:
        raise ValueError
    if control_ticker_symbol is None:
        control_ticker_symbol = ""
    if control_weight > 0 and control_ticker_symbol == "":
        raise ValueError
    weight = d("1") - control_weight
    return {
        **({control_ticker_symbol: control_weight} if control_weight > 0 else {}),
        **({ticker_symbol: weight} if weight > 0 else {}),
    }


def unround(f: Callable[[int], tuple[d, d]]) -> Callable[[int], tuple[d, d]]:
    @wraps(f)
    def wrapper(percentage: int) -> tuple[d, d]:
        if not 0 <= percentage <= 100:
            raise ValueError
        min_, max_ = f(percentage)
        return max(min_, d("0")), min(max_, d("100"))

    return wrapper


@unround
def unround_asset_allocation(percentage: int) -> tuple[d, d]:
    return percentage - d("0.5"), percentage + d("0.5")


@unround
def unround_style_box(percentage: int) -> tuple[d, d]:
    if percentage % 2 != 0:
        return percentage - 1 + d("0.5050"), percentage + 1 - d("0.5050")
    return percentage - d("0.5050"), percentage + d("0.5050")


def round3(x: d, rounding, base: Optional[d] = None) -> d:
    if base is None:
        base = d("1")
    with localcontext() as ctx:
        ctx.rounding = rounding

        # Decimal.__round__ ignores ctx.rounding without the 2nd argument
        return base * round(x / base, 0)


def solve(
    control_weight: d,
    portfolio_percentage: d,
    control_percentage: int,
    equity_percentage: Optional[d] = None,
) -> d:
    if equity_percentage is None:
        equity_percentage = d("100")
    if not 0 <= control_weight < 1:
        raise ValueError
    if not 0 <= portfolio_percentage <= 100:
        raise ValueError
    if not 0 <= control_percentage <= 100:
        raise ValueError
    equity_weight = equity_percentage / 100
    #  p := percentage
    # cw := control_weight
    # pp := portfolio_percentage
    # cp := control_percentage
    # ew := equity_weight
    #
    # ew = ep / 100
    # pp = ((1 - cw) ew p + cw cp) / ((1 - cw) ew + cw)
    # => (1 - cw) ew p + cw cp = pp ((1 - cw) ew + cw)
    # => (1 - cw) ew p = pp ((1 - cw) ew + cw) - cw cp
    # => p = (pp ((1 - cw) ew + cw) - cw cp) / ((1 - cw) ew)
    return (
        portfolio_percentage * ((1 - control_weight) * equity_weight + control_weight)
        - control_weight * control_percentage
    ) / ((1 - control_weight) * equity_weight)


class SearchError(RuntimeError):
    pass


class Morningstar:
    _config: ConfigParser
    _credentials: ConfigParser
    _driver_cm: ContextManagerWrapper[Firefox]
    _addons_directory_cm: ContextManagerWrapper[TemporaryDirectory]
    _headless: Optional[bool]
    _cache: dict[tuple[tuple[str, d], ...], dict[str, Any]]
    _morningstar_firefox: MorningstarWebDriver
    _control_ticker_symbols: tuple[str, ...]

    def _init(self) -> None:
        for try_ in count():
            try:
                self._morningstar_firefox = MorningstarWebDriver(
                    self._credentials,
                    self._cache,
                    self._config,
                    self._driver_cm.value,
                    self._addons_directory_cm.value,
                )
                return
            except WebDriverException as e:
                if try_ >= int(self._config["morningstar_firefox.init_try"]["try.max_retries"]):
                    raise RuntimeError from e
                self._driver_cm.new(get_driver(self._config, self._headless))
                self._addons_directory_cm.new(get_addons_directory())

    @staticmethod
    def _retry(
        section_name: str,
    ) -> Callable[
        [Callable[["Morningstar", dict[str, d]], Return]],
        #
        Callable[["Morningstar", dict[str, d]], Return],
    ]:
        def decorator(
            f: Callable[["Morningstar", dict[str, d]], Return]
        ) -> Callable[["Morningstar", dict[str, d]], Return]:
            @wraps(f)
            def wrapper(self, portfolio: dict[str, d]) -> Return:
                for try_ in count():
                    try:
                        return f(self, portfolio)
                    except WebDriverException as e:
                        if try_ >= int(self._config[section_name]["try.max_retries"]):
                            raise RuntimeError from e
                        self._init()
                assert False

            return wrapper

        return decorator

    @_retry("morningstar_firefox.asset_allocation_try")
    def asset_allocation(self, portfolio: dict[str, d]) -> tuple[int, ...]:
        return self._morningstar_firefox.asset_allocation(portfolio)

    @_retry("morningstar_firefox.style_box_try")
    def style_box(self, portfolio: dict[str, d]) -> tuple[int, ...]:
        return self._morningstar_firefox.style_box(portfolio)

    def __init__(
        self,
        config: ConfigParser,
        credentials: ConfigParser,
        driver_cm: ContextManagerWrapper[Firefox],
        addons_directory_cm: ContextManagerWrapper[TemporaryDirectory],
        lv: str,
        lb: str,
        lg: str,
        mv: str,
        mb: str,
        mg: str,
        sv: str,
        sb: str,
        sg: str,
        *args,
        headless: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._config = config
        self._credentials = credentials
        self._driver_cm = driver_cm
        self._addons_directory_cm = addons_directory_cm
        self._headless = headless
        self._cache = {}

        self._init()

        control_ticker_symbols = (lv, lb, lg, mv, mb, mg, sv, sb, sg)
        for i, ticker_symbol in enumerate(control_ticker_symbols):
            expected_style_box = tuple(100 if j == i else 0 for j in range(9))
            style_box = self.style_box({ticker_symbol: d("1")})
            if style_box != expected_style_box:
                raise RuntimeError
        self._control_ticker_symbols = control_ticker_symbols

    def _search(
        self,
        f: Callable[["Morningstar", dict[str, d]], int],
        unround: Callable[[int], tuple[d, d]],
        control_ticker_symbol_lt50: str,
        control_ticker_symbol_ge50: str,
        solve: Callable[[d, d, int], d],
        ticker_symbol: str,
        *,
        _test_inject_after: Optional[int] = None,
        _test_fault_percentage: Optional[int] = None,
    ) -> tuple[d, d, int]:
        if _test_inject_after is None:
            _test_inject_after = -1
        if _test_fault_percentage is None:
            _test_fault_percentage = 100

        lb = d("0")
        rounded_percentage = f(
            self,
            #
            dummy_portfolio(ticker_symbol),
        )
        percentage_lb, percentage_ub = unround(rounded_percentage)
        if rounded_percentage < 50:
            control_ticker_symbol = control_ticker_symbol_lt50
        else:
            control_ticker_symbol = control_ticker_symbol_ge50
        ub = d("1")
        control_percentage = f(
            self,
            #
            dummy_portfolio(ticker_symbol, control_ticker_symbol, ub),
        )
        target_portfolio_percentage = rounded_percentage + round3(
            d(control_percentage - rounded_percentage) / 2, ROUND_UP
        )
        lb_portfolio_percentage = rounded_percentage
        ub_portfolio_percentage = control_percentage
        for _test_j in count():
            control_weight = round3((lb + ub) / 2, ROUND_UP, base=d("0.000001"))
            if control_weight == ub:
                if abs(ub_portfolio_percentage - lb_portfolio_percentage) != 1:
                    raise SearchError
                return (
                    ub,
                    unround(ub_portfolio_percentage)[0 if lb_portfolio_percentage < ub_portfolio_percentage else 1],
                    control_percentage,
                )
            portfolio_percentage = f(
                self,
                #
                dummy_portfolio(ticker_symbol, control_ticker_symbol, control_weight),
            )

            if _test_j == _test_inject_after:
                portfolio_percentage = _test_fault_percentage

            if target_portfolio_percentage > rounded_percentage:
                if not lb_portfolio_percentage <= portfolio_percentage <= ub_portfolio_percentage:
                    raise SearchError
            else:
                if not ub_portfolio_percentage <= portfolio_percentage <= lb_portfolio_percentage:
                    raise SearchError
            new_percentage_lb = solve(
                #
                *(control_weight, unround(portfolio_percentage)[0], control_percentage)
            )
            percentage_lb = max(percentage_lb, new_percentage_lb)
            new_percentage_ub = solve(
                #
                *(control_weight, unround(portfolio_percentage)[1], control_percentage)
            )
            percentage_ub = min(percentage_ub, new_percentage_ub)
            if percentage_ub < percentage_lb:
                raise SearchError
            if (
                d(portfolio_percentage - target_portfolio_percentage)
                / d(target_portfolio_percentage - rounded_percentage)
                >= 0
            ):
                # already hit target_portfolio_percentage
                #
                # example 1: rounded_percentage=  20, control_percentage=   0 => target_portfolio_percentage=  10
                #
                #     portfolio_percentage=  11: (  11-  10)/(  10-  20)=  1/ -10<0
                #     portfolio_percentage=  10: (  10-  10)/(  10-  20)=  0/ -10=0
                #     portfolio_percentage=   9: (   9-  10)/(  10-  20)= -1/ -10>0
                #
                #
                # example 2: rounded_percentage=  40, control_percentage= 100 => target_portfolio_percentage=  70
                #
                #     portfolio_percentage=  69: (  69-  70)/(  70-  40)= -1/  30<0
                #     portfolio_percentage=  70: (  70-  70)/(  70-  40)=  0/  30=0
                #     portfolio_percentage=  71: (  71-  70)/(  70-  40)=  1/  30>0
                #
                ub = control_weight
                ub_portfolio_percentage = portfolio_percentage
            else:
                lb = control_weight
                lb_portfolio_percentage = portfolio_percentage
        assert False

    @staticmethod
    def _retry_search(
        section_name: str,
    ) -> Callable[
        [
            Callable[
                [
                    "Morningstar",
                    str,
                    int,
                    #
                    DefaultNamedArg(Optional[int], "_test_inject_after"),
                    DefaultNamedArg(Optional[int], "_test_fault_percentage"),
                ],
                tuple[d, d, int],
            ]
        ],
        #
        Callable[
            [
                "Morningstar",
                str,
                int,
                #
                DefaultNamedArg(Optional[Iterable[Optional[int]]], "_test_inject_after"),
                DefaultNamedArg(Optional[Iterable[Optional[int]]], "_test_fault_percentage"),
            ],
            tuple[d, d, int],
        ],
    ]:
        def decorator(
            f: Callable[
                [
                    "Morningstar",
                    str,
                    int,
                    #
                    DefaultNamedArg(Optional[int], "_test_inject_after"),
                    DefaultNamedArg(Optional[int], "_test_fault_percentage"),
                ],
                tuple[d, d, int],
            ]
        ) -> Callable[
            [
                "Morningstar",
                str,
                int,
                #
                DefaultNamedArg(Optional[Iterable[Optional[int]]], "_test_inject_after"),
                DefaultNamedArg(Optional[Iterable[Optional[int]]], "_test_fault_percentage"),
            ],
            tuple[d, d, int],
        ]:
            @wraps(f)
            def wrapper(
                self,
                ticker_symbol: str,
                i: int,
                *,
                _test_inject_after: Optional[Iterable[Optional[int]]] = None,
                _test_fault_percentage: Optional[Iterable[Optional[int]]] = None,
            ) -> tuple[d, d, int]:
                if _test_inject_after is None:
                    _test_inject_after = repeat(None)
                else:
                    _test_inject_after = chain(_test_inject_after, repeat(None))
                if _test_fault_percentage is None:
                    _test_fault_percentage = repeat(None)
                else:
                    _test_fault_percentage = chain(_test_fault_percentage, repeat(None))

                for try_, _test_inject_after_try, _test_fault_percentage_try in zip(
                    count(), _test_inject_after, _test_fault_percentage
                ):
                    try:
                        return f(
                            self,
                            ticker_symbol,
                            i,
                            _test_inject_after=_test_inject_after_try,
                            _test_fault_percentage=_test_fault_percentage_try,
                        )
                    except SearchError as e:
                        if try_ >= int(self._config[section_name]["try.max_retries"]):
                            raise RuntimeError from e
                        self._driver_cm.new(get_driver(self._config, self._headless))
                        self._addons_directory_cm.new(get_addons_directory())
                        self._cache = {}

                        self._init()
                assert False

            return wrapper

        return decorator

    @_retry_search("morningstar_firefox.asset_allocation_search_try")
    def search_asset_allocation(
        self,
        ticker_symbol: str,
        i: int,
        *,
        _test_inject_after: Optional[int] = None,
        _test_fault_percentage: Optional[int] = None,
    ) -> tuple[d, d, int]:
        if i not in (1, 2):
            raise ValueError
        control_ticker_symbols = {
            1: self._control_ticker_symbols[1]
            if ticker_symbol == self._control_ticker_symbols[0]
            else self._control_ticker_symbols[0],
            2: "TSM",  # XXX
        }
        return self._search(
            lambda self, portfolio: self.asset_allocation(portfolio)[i],
            unround_asset_allocation,
            control_ticker_symbols[i],
            tuple(
                control_ticker_symbol
                for j, control_ticker_symbol in control_ticker_symbols.items()
                #
                if j != i
            )[0],
            lambda control_weight, portfolio_percentage, control_percentage: solve(
                control_weight, portfolio_percentage, control_percentage
            ),
            ticker_symbol,
            _test_inject_after=_test_inject_after,
            _test_fault_percentage=_test_fault_percentage,
        )

    @_retry_search("morningstar_firefox.style_box_search_try")
    def search_style_box(
        self,
        ticker_symbol: str,
        i: int,
        *,
        _test_inject_after: Optional[int] = None,
        _test_fault_percentage: Optional[int] = None,
    ) -> tuple[d, d, int]:
        equity_percentage = (
            #
            solve(*self.search_asset_allocation(ticker_symbol, 1))
            + solve(*self.search_asset_allocation(ticker_symbol, 2))
        )
        return self._search(
            lambda self, portfolio: self.style_box(portfolio)[i],
            unround_style_box,
            self._control_ticker_symbols[i],
            (self._control_ticker_symbols[:i] + self._control_ticker_symbols[i + 1 :])[0],
            lambda control_weight, portfolio_percentage, control_percentage: solve(
                control_weight, portfolio_percentage, control_percentage, equity_percentage
            ),
            ticker_symbol,
            _test_inject_after=_test_inject_after,
            _test_fault_percentage=_test_fault_percentage,
        )

    def fund(
        self,
        ticker_symbol: str,
        *,
        _test_inject_after: Optional[dict[str, Optional[Iterable[Optional[int]]]]] = None,
        _test_fault_percentage: Optional[dict[str, Optional[Iterable[Optional[int]]]]] = None,
    ) -> ndarray:
        if _test_inject_after is None:
            _test_inject_after = defaultdict(lambda: None)
        else:
            _test_inject_after = defaultdict(lambda: None, _test_inject_after)
        if _test_fault_percentage is None:
            _test_fault_percentage = defaultdict(lambda: None)
        else:
            _test_fault_percentage = defaultdict(lambda: None, _test_fault_percentage)

        return array(
            tuple(
                float(
                    solve(
                        *self.search_style_box(
                            ticker_symbol,
                            i,
                            _test_inject_after=_test_inject_after[str(i)],
                            _test_fault_percentage=_test_fault_percentage[str(i)],
                        )
                    )
                )
                for i in range(9)
            )
        ).reshape(3, 3)


CONFIG_PATH = "config.ini"
CREDENTIALS_PATH = "credentials.ini"


def init(
    *,
    config_path: Optional[str] = None,
    credentials_path: Optional[str] = None,
    headless: Optional[bool] = None,
) -> tuple[ConfigParser, ConfigParser, ContextManagerWrapper[Firefox], ContextManagerWrapper[TemporaryDirectory]]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_PATH)
    config = ConfigParser()
    config.read(config_path)
    if credentials_path is None:
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CREDENTIALS_PATH)
    credentials = ConfigParser()
    credentials.read(credentials_path)

    driver_cm = ContextManagerWrapper(get_driver(config, headless))
    addons_directory_cm = ContextManagerWrapper(get_addons_directory())
    return config, credentials, driver_cm, addons_directory_cm


def main(
    lv: str,
    lb: str,
    lg: str,
    mv: str,
    mb: str,
    mg: str,
    sv: str,
    sb: str,
    sg: str,
    ticker_symbols: Iterable[str],
    *,
    config_path: Optional[str] = None,
    credentials_path: Optional[str] = None,
    headless: Optional[bool] = None,
    _test_inject_after: Optional[dict[str, Optional[dict[str, Optional[Iterable[Optional[int]]]]]]] = None,
    _test_fault_percentage: Optional[dict[str, Optional[dict[str, Optional[Iterable[Optional[int]]]]]]] = None,
) -> dict[str, ndarray]:
    if _test_inject_after is None:
        _test_inject_after = defaultdict(lambda: None)
    else:
        _test_inject_after = defaultdict(lambda: None, _test_inject_after)
    if _test_fault_percentage is None:
        _test_fault_percentage = defaultdict(lambda: None)
    else:
        _test_fault_percentage = defaultdict(lambda: None, _test_fault_percentage)

    config, credentials, driver_cm, addons_directory_cm = init(
        config_path=config_path,
        credentials_path=credentials_path,
        headless=headless,
    )
    with driver_cm, addons_directory_cm:
        morningstar = Morningstar(
            config,
            credentials,
            driver_cm,
            addons_directory_cm,
            lv,
            lb,
            lg,
            mv,
            mb,
            mg,
            sv,
            sb,
            sg,
            headless=headless,
        )

        return {
            ticker_symbol: morningstar.fund(
                ticker_symbol,
                _test_inject_after=_test_inject_after[ticker_symbol],
                _test_fault_percentage=_test_fault_percentage[ticker_symbol],
            )
            for ticker_symbol in ticker_symbols
        }


def write(funds_path: str, funds: dict[str, ndarray]) -> None:
    Path(os.path.dirname(funds_path)).mkdir(parents=True, exist_ok=True)
    with open(funds_path, "w", newline="") as funds_file:
        writer = csv.writer(funds_file)
        writer.writerows(
            ((ticker_symbol, *style_box.flatten().astype(str)) for ticker_symbol, style_box in funds.items())
        )


if __name__ == "__main__":
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config")
    parser.add_argument("--credentials")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("lv")
    parser.add_argument("lb")
    parser.add_argument("lg")
    parser.add_argument("mv")
    parser.add_argument("mb")
    parser.add_argument("mg")
    parser.add_argument("sv")
    parser.add_argument("sb")
    parser.add_argument("sg")
    parser.add_argument("csv")
    parser.add_argument("funds", nargs="*")
    parser.add_argument("--test-inject-after", type=json.loads)
    parser.add_argument("--test-fault-percentage", type=json.loads)
    args = parser.parse_args()

    funds = main(
        args.lv,
        args.lb,
        args.lg,
        args.mv,
        args.mb,
        args.mg,
        args.sv,
        args.sb,
        args.sg,
        args.funds,
        config_path=args.config,
        credentials_path=args.credentials,
        headless=args.headless,
        _test_inject_after=args.test_inject_after,
        _test_fault_percentage=args.test_fault_percentage,
    )

    write(args.csv, funds)
