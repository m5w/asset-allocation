# Copyright (C) 2022 Matthew Marting
# SPDX-License-Identifier: GPL-3.0-or-later

from argparse import ArgumentParser
from collections import deque
from configparser import ConfigParser
from contextlib import AbstractContextManager
import csv
from decimal import Decimal as d
from decimal import localcontext
from decimal import ROUND_UP
from functools import cache
from itertools import count
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory
import lxml.html
from numpy import array
import requests
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager


class MorningstarWebdriver:
    def _addon_url(self, addon):
        webpage_url = f"https://addons.mozilla.org/en-US/firefox/addon/{addon}"
        webpage_response = requests.get(
            webpage_url,
            #
            timeout=float(self._config["morningstar.webdriver"]["addon_webpage_timeout"]),
        )
        webpage_response.raise_for_status()
        webpage_html = lxml.html.fromstring(webpage_response.text)
        download_file = webpage_html.xpath(
            #
            "//a[contains(descendant-or-self::*, 'Download file')]"
        )[0]
        return download_file.attrib["href"]

    def __init__(self, config, credentials, driver, addons_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._driver = driver
        for _, addon in self._config.items("webdriver.addons"):
            url = self._addon_url(addon)
            response = requests.get(
                url,
                #
                timeout=float(self._config["morningstar.webdriver"]["addon_timeout"]),
            )
            response.raise_for_status()
            basename = os.path.basename(url)
            path = os.path.join(addons_directory, basename)
            with open(path, "wb") as file:
                file.write(response.content)
            self._driver.install_addon(path, temporary=True)

        self._driver.get(
            #
            "https://www.morningstar.com/auth-init?"
            "destination=https://www.morningstar.com/instant-x-ray"
        )
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["email_input_clickable_timeout"]),
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
            float(self._config["morningstar.webdriver"]["password_input_clickable_timeout"]),
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
            float(self._config["morningstar.webdriver"]["sign_in_clickable_timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//button[contains(descendant-or-self::*, 'Sign In')]")
            )
        ).click()

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["instant_x_ray_iframe_timeout"]),
        ).until(
            EC.frame_to_be_available_and_switch_to_it(
                #
                (By.ID, "instant-x-ray")
            )
        )

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["holding_value_clickable_timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Holding Value')]")
            )
        ).click()
        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["percentage_value_clickable_timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Percentage Value %')]")
            )
        ).click()

        self._instant_x_ray = False

    _FIELD_COUNT = 10

    def _show_instant_x_ray(self, portfolio):
        assert not self._instant_x_ray

        reset, t0 = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["reset_clickable_timeout"]),
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
            float(self._config["morningstar.webdriver"]["t0_stale_timeout"]),
        ).until(
            EC.staleness_of(
                #
                t0
            )
        )

        entry_fields = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["entry_fields_clickable_timeout"]),
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
            float(self._config["morningstar.webdriver"]["show_instant_x_ray_clickable_timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Show Instant X-Ray')]")
            )
        ).send_keys(
            Keys.ENTER
        )

        self._instant_x_ray = True

    def _style_box(self):
        assert self._instant_x_ray

        style_box = WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["style_box_visible_timeout"]),
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

    def _edit_holdings(self):
        assert self._instant_x_ray

        WebDriverWait(
            self._driver,
            #
            float(self._config["morningstar.webdriver"]["edit_holdings_clickable_timeout"]),
        ).until(
            EC.element_to_be_clickable(
                #
                (By.XPATH, "//a[contains(descendant-or-self::*, 'Edit Holdings')]")
            )
        ).send_keys(
            Keys.ENTER
        )

        self._instant_x_ray = False

    def style_box(self, portfolio):
        if self._instant_x_ray:
            self._edit_holdings()
        self._show_instant_x_ray(portfolio)
        return self._style_box()


class Cache:
    @cache
    def _style_box_cache(self, hashable_portfolio):
        portfolio = {ticker_symbol: weight for ticker_symbol, weight in hashable_portfolio}
        return self._morningstar_driver.style_box(portfolio)

    def style_box(self, morningstar_driver, portfolio):
        self._morningstar_driver = morningstar_driver
        hashable_portfolio = tuple(sorted(portfolio.items()))
        return self._style_box_cache(hashable_portfolio)


def dummy_portfolio(ticker_symbol, control_ticker_symbol=None, control_weight=None):
    if control_weight is None:
        control_weight = d("0")
    if control_weight < 0:
        raise ValueError
    if control_weight > 0 and control_ticker_symbol is None:
        raise ValueError
    return {
        **({control_ticker_symbol: control_weight} if control_weight > 0 else {}),
        ticker_symbol: d("1") - control_weight,
    }


def round3(x, rounding, base=None):
    if base is None:
        base = d("1")
    with localcontext() as ctx:
        ctx.rounding = rounding

        # Decimal.__round__ ignores ctx.rounding without the 2nd argument
        return base * round(x / base, 0)


def solve(control_weight, percentage, control_percentage):
    # percentage = (1 - control_weight) start_percentage + control_weight control_percentage
    # => (1 - control_weight) start_percentage = percentage - control_weight control_percentage
    # => start_percentage = (percentage - control_weight control_percentage) / (1 - control_weight)
    return (percentage - control_weight * control_percentage) / (1 - control_weight)


class ContextManagerWrapper(AbstractContextManager):
    def __init__(self, thing):
        self.thing = thing

    def __enter__(self):
        self.value = self.thing.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.value
        return self.thing.__exit__(exc_type, exc_value, traceback)

    def new(self, thing):
        self.__exit__(None, None, None)
        self.thing = thing
        self.__enter__()


def get_driver(headless):
    options = Options()
    options.headless = headless
    return Firefox(options=options, service=Service(GeckoDriverManager().install()))


def get_addons_directory():
    return TemporaryDirectory()


class Morningstar:
    def _init(self):
        for i in count():
            try:
                self._morningstar_driver = MorningstarWebdriver(
                    self._config, self._credentials, self._driver_cm.value, self._addons_directory_cm.value
                )
                return
            except WebDriverException as e:
                if i >= int(self._config["morningstar.webdriver"]["max_tries"]) - 1:
                    raise RuntimeError from e
                self._driver_cm.new(get_driver(self._headless))
                self._addons_directory_cm.new(get_addons_directory())

    def style_box(self, portfolio):
        for i in count():
            try:
                return self._cache.style_box(self._morningstar_driver, portfolio)
            except WebDriverException as e:
                if i >= int(self._config["morningstar.webdriver"]["max_tries"]) - 1:
                    raise RuntimeError from e
                self._init()

    def __init__(
        self,
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
        *args,
        headless=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if headless is None:
            headless = False
        self._config = config
        self._credentials = credentials
        self._driver_cm = driver_cm
        self._addons_directory_cm = addons_directory_cm
        self._headless = headless

        self._init()

        self._cache = Cache()

        control_ticker_symbols = (lv, lb, lg, mv, mb, mg, sv, sb, sg)
        for i, ticker_symbol in enumerate(control_ticker_symbols):
            expected_style_box = tuple(100 if j == i else 0 for j in range(9))
            style_box = self.style_box({ticker_symbol: 1})
            if style_box != expected_style_box:
                raise RuntimeError
        self._control_ticker_symbols = control_ticker_symbols

    def _search(self, ticker_symbol, i):
        lb = d("0")
        ub = d("1")
        start_percentage = self.style_box(
            #
            dummy_portfolio(ticker_symbol)
        )[i]
        if start_percentage < 50:
            control_ticker_symbol = self._control_ticker_symbols[i]
        else:
            control_ticker_symbol = (self._control_ticker_symbols[:i] + self._control_ticker_symbols[i + 1 :])[0]
        control_percentage = self.style_box(
            #
            dummy_portfolio(ticker_symbol, control_ticker_symbol, ub)
        )[i]
        if control_percentage == start_percentage:
            return lb, d(start_percentage), control_percentage
        target_percentage = start_percentage + round3(d(control_percentage - start_percentage) / 2, ROUND_UP)
        lb_percentage = start_percentage
        ub_percentage = control_percentage
        while True:
            control_weight = round3((lb + ub) / 2, ROUND_UP, base=d("0.000001"))
            if control_weight == ub:
                if ub_percentage % 2 != 0:
                    return lb, lb_percentage + (ub_percentage - lb_percentage) * d("0.5050"), control_percentage
                return ub, ub_percentage + (lb_percentage - ub_percentage) * d("0.5050"), control_percentage
            percentage = self.style_box(
                #
                dummy_portfolio(ticker_symbol, control_ticker_symbol, control_weight)
            )[i]
            if d(percentage - target_percentage) / d(target_percentage - start_percentage) >= 0:
                # already hit target_percentage
                #
                # example 1: start_percentage=  20, control_percentage=   0 => target_percentage=  10
                #
                #     percentage=  11: (  11-  10)/(  10-  20)=  1/ -10<0
                #     percentage=  10: (  10-  10)/(  10-  20)=  0/ -10=0
                #     percentage=   9: (   9-  10)/(  10-  20)= -1/ -10>0
                #
                #
                # example 2: start_percentage=  40, control_percentage= 100 => target_percentage=  70
                #
                #     percentage=  69: (  69-  70)/(  70-  40)= -1/  30<0
                #     percentage=  70: (  70-  70)/(  70-  40)=  0/  30=0
                #     percentage=  71: (  71-  70)/(  70-  40)=  1/  30>0
                #
                ub = control_weight
                ub_percentage = percentage
            else:
                lb = control_weight
                lb_percentage = percentage

    def fund(self, ticker_symbol):
        return array(tuple(float(solve(*self._search(ticker_symbol, i))) for i in range(9))).reshape(3, 3)


CONFIG_PATH = "config.ini"
CREDENTIALS_PATH = "credentials.ini"


def main(
    lv,
    lb,
    lg,
    mv,
    mb,
    mg,
    sv,
    sb,
    sg,
    ticker_symbols,
    *,
    config_path=None,
    credentials_path=None,
    headless=None,
):
    if config_path is None:
        config_path = CONFIG_PATH
    if credentials_path is None:
        credentials_path = CREDENTIALS_PATH
    config = ConfigParser()
    config.read(config_path)
    credentials = ConfigParser()
    credentials.read(credentials_path)
    driver_cm = ContextManagerWrapper(get_driver(headless))
    addons_directory_cm = ContextManagerWrapper(get_addons_directory())
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
        )

        return {ticker_symbol: morningstar.fund(ticker_symbol) for ticker_symbol in ticker_symbols}


def write(funds_path, funds):
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
    )

    write(args.csv, funds)
