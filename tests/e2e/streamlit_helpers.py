"""Streamlit widget interaction helpers for Playwright E2E tests.

All CSS selectors are isolated here so Streamlit version upgrades
only require changes in this one file.
"""

from __future__ import annotations

import re

from playwright.sync_api import Page

# ── Selector Constants ───────────────────────────────────────────────

SIDEBAR = 'section[data-testid="stSidebar"]'
MAIN = '[data-testid="stAppViewContainer"]'
ALERT = '[data-testid="stAlert"]'

# Whether a rerun is in flight is read off the app root, which Streamlit stamps
# with data-test-script-state ("initial" / "running" / "notRunning"). The
# stStatusWidget this used to watch is no longer rendered at all, so waiting for
# it to appear and then vanish returned instantly every time. Long operations
# then had their results asserted while the script was still running, which is
# what made the scanner, factors, portfolio and live tests fail intermittently.
SCRIPT_RUNNING = '[data-testid="stApp"][data-test-script-state="running"]'
SCRIPT_IDLE = (
    '[data-testid="stApp"][data-test-script-state="notRunning"], '
    '[data-testid="stApp"][data-test-script-state="initial"]'
)
DATAFRAME = '[data-testid="stDataFrame"]'
PLOTLY_CHART = ".stPlotlyChart, .js-plotly-plot"
# st.metric elements (standard) + custom HTML metric cards
METRIC = '[data-testid="stMetric"]'
METRIC_CARD = ".metric-card"
METRIC_CARD_LABEL = ".metric-card-label"
METRIC_CARD_VALUE = ".metric-card-value"

# Streamlit 1.60 moved its widgets from BaseWeb to react-aria. The outer
# ``data-testid`` containers survived that move, the insides did not:
# ``[data-baseweb="select"]`` and the slider's ``[role="slider"]`` thumb are both
# gone. These constants name the replacements.
#
# The selectbox dropdown renders into a body-level portal, NOT inside the widget,
# so option locators must be page-scoped. It is also virtualised: options sit at
# absolute offsets and Playwright refuses to click them ("element is outside of
# the viewport"), even with force=True. Type-to-filter then Enter is the only
# reliable way to drive it.
SEL_COMBOBOX_INPUT = 'input[role="combobox"]'
SEL_COMBOBOX_TRIGGER = 'button[aria-haspopup="listbox"]'
SEL_DROPDOWN = '[data-testid="stSelectboxVirtualDropdown"]'
SEL_OPTION = '[role="option"]'
# The slider thumb is a visually hidden native range input, which is better than
# the old ARIA attributes: it carries real min/max/step/value.
SEL_SLIDER_INPUT = 'input[type="range"]'
SEL_SLIDER_THUMB_VALUE = '[data-testid="stSliderThumbValue"]'

# The symbol picker is st_searchbox, a custom component, so it lives in its own
# iframe and none of the ordinary Streamlit widget selectors reach it.
SEL_SEARCHBOX = 'iframe[title="streamlit_searchbox.searchbox"]'
# Date ranges are chosen with a row of preset buttons keyed "{prefix}_dp_{name}",
# not with st.date_input.
DATE_PRESETS = ("1W", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y")

# Widget container selectors
SEL_SELECTBOX = '[data-testid="stSelectbox"]'
SEL_MULTISELECT = '[data-testid="stMultiSelect"]'
SEL_SLIDER = '[data-testid="stSlider"]'
SEL_CHECKBOX = '[data-testid="stCheckbox"]'
SEL_TOGGLE = '[data-testid="stToggle"]' if True else '.stToggle'
SEL_BUTTON = '[data-testid="stButton"]'
SEL_TEXT_INPUT = '[data-testid="stTextInput"]'
SEL_NUMBER_INPUT = '[data-testid="stNumberInput"]'
SEL_TEXT_AREA = '[data-testid="stTextArea"]'
SEL_DATE_INPUT = '[data-testid="stDateInput"]'
SEL_FILE_UPLOADER = '[data-testid="stFileUploader"]'
SEL_EXPANDER = '[data-testid="stExpander"]'
SEL_RADIO = '[data-testid="stRadio"]'

# Sidebar navigation is a column of st.button widgets keyed ``nav_<page name>``.
# Streamlit stamps each key onto the element container as an ``st-key-`` class,
# sanitising every non-alphanumeric character to a hyphen, so the key
# ``nav_📡 Market Scanner`` renders as ``st-key-nav_---Market-Scanner``. Matching
# that prefix selects exactly the ten nav buttons and skips the watchlist,
# footer and page-specific controls that share the sidebar.
NAV_BUTTON = '[class*="st-key-nav_"]'
# Not-yet-built pages carry this marker at the end of their button label.
SCAFFOLD_MARKER = "◦"


def _plain_label(page_name: str) -> str:
    """Strip the leading emoji and any trailing scaffold marker from a page name."""
    return re.sub(r"^[^\w]+", "", page_name).replace(SCAFFOLD_MARKER, "").strip()

# Timeout defaults (ms)
RERUN_TIMEOUT = 30_000
WIDGET_TIMEOUT = 10_000
LONG_OP_TIMEOUT = 60_000


class StreamlitPage:
    """Wrapper around a Playwright Page with Streamlit-aware helpers."""

    def __init__(self, page: Page) -> None:
        self.page = page

    # ── Navigation ───────────────────────────────────────────────────

    def navigate_to(self, page_name: str) -> None:
        """Click a sidebar navigation button to navigate to a page.

        ``page_name`` should be the full display name including emoji,
        e.g. "📡 Market Scanner".
        """
        buttons = self.page.locator(SIDEBAR).locator(NAV_BUTTON)
        # filter(has_text=...) matches on rendered text, so the emoji needs no
        # CSS escaping the way it would inside a :has-text() selector string.
        option = buttons.filter(has_text=page_name)
        if option.count() == 0:
            # Fallback: match the plain label, without the emoji.
            option = buttons.filter(has_text=_plain_label(page_name))
        option.first.locator("button").first.click()
        self.wait_for_rerun()

    def get_page_names(self) -> list[str]:
        """Return all sidebar navigation button labels."""
        buttons = self.page.locator(SIDEBAR).locator(NAV_BUTTON)
        return [
            buttons.nth(i).inner_text().replace(SCAFFOLD_MARKER, "").strip()
            for i in range(buttons.count())
        ]

    # ── Wait Helpers ─────────────────────────────────────────────────

    def wait_for_rerun(self, timeout: int = RERUN_TIMEOUT) -> None:
        """Block until Streamlit's script has finished running.

        A rerun that outlasts *timeout* raises rather than returning quietly.
        Swallowing it is what produced the "neither results nor a user-facing
        message appeared" failures: the wait gave up, the assertion sampled a
        half-drawn page, and the report blamed the app instead of the clock.
        """
        try:
            # A fast rerun can finish before this look, so not seeing "running"
            # is fine. The wait below is the one that matters.
            self.page.wait_for_selector(
                SCRIPT_RUNNING, state="attached", timeout=2_000
            )
        except Exception:
            pass

        self.page.wait_for_selector(SCRIPT_IDLE, state="attached", timeout=timeout)
        # React paints the new elements a frame after the state flips, and
        # get_visible_text/get_metric_count read the DOM with no patience of
        # their own.
        self.page.wait_for_timeout(250)

    def wait_for_app_loaded(self) -> None:
        """Wait for the initial Streamlit app shell to render."""
        self.page.wait_for_selector(MAIN, timeout=15_000)

    # ── Widget Finders ───────────────────────────────────────────────

    def _scope(self, sidebar: bool):
        """Return sidebar or main content locator."""
        return self.page.locator(SIDEBAR if sidebar else MAIN)

    def _find_widget_by_label(
        self, selector: str, label: str, sidebar: bool = False
    ):
        """Find a widget container by its label text."""
        scope = self._scope(sidebar)
        # Streamlit puts labels as the first child div in the widget container
        widgets = scope.locator(selector)
        for i in range(widgets.count()):
            widget = widgets.nth(i)
            if label.lower() in widget.inner_text().lower():
                return widget
        # Fallback: broader text match
        return scope.locator(f'{selector}:has-text("{label}")').first

    # ── Selectbox ────────────────────────────────────────────────────

    def _open_combobox(self, widget) -> None:
        """Open a selectbox dropdown, whichever of its two hit targets works.

        Clicking the combobox input opens the list on most pages; on some the
        trigger button is the one that responds. Which one works varies with
        where the widget sits, so try the input, then fall back to the trigger.
        """
        if self.page.locator(SEL_DROPDOWN).count() > 0:
            return
        for target in (SEL_COMBOBOX_INPUT, SEL_COMBOBOX_TRIGGER):
            el = widget.locator(target)
            if el.count() == 0:
                continue
            el.first.click()
            try:
                self.page.wait_for_selector(SEL_DROPDOWN, timeout=3_000)
                return
            except Exception:
                continue

    def select_box(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Choose *value* in a Streamlit selectbox.

        Types the value to filter the list and commits with Enter. Clicking the
        option is not an option: the dropdown is virtualised, so Playwright sees
        every option as outside the viewport and refuses the click, even with
        force=True.
        """
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        self._open_combobox(widget)
        combo = widget.locator(SEL_COMBOBOX_INPUT).first
        combo.fill(value)
        # Wait for the list to narrow to the typed value before committing.
        self.page.wait_for_selector(SEL_OPTION, timeout=WIDGET_TIMEOUT)
        self.page.keyboard.press("Enter")
        self.wait_for_rerun()

    def get_selectbox_value(
        self, label: str, sidebar: bool = False
    ) -> str:
        """Read the current value of a selectbox."""
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        return widget.locator(SEL_COMBOBOX_INPUT).first.input_value().strip()

    def get_selectbox_options(
        self, label: str, sidebar: bool = False
    ) -> list[str]:
        """Open a selectbox and return the option texts, then close it.

        The dropdown is virtualised, so for a long list this returns only the
        options currently rendered, not every option in the widget.
        """
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        self._open_combobox(widget)
        self.page.wait_for_selector(SEL_OPTION, timeout=WIDGET_TIMEOUT)
        options_loc = self.page.locator(SEL_OPTION)
        opts = [
            options_loc.nth(i).inner_text()
            for i in range(options_loc.count())
        ]
        # Close by pressing Escape
        self.page.keyboard.press("Escape")
        return opts

    # ── Multiselect ──────────────────────────────────────────────────

    def multiselect(
        self, label: str, values: list[str], sidebar: bool = False
    ) -> None:
        """Set multiselect values (clears existing, adds new)."""
        widget = self._find_widget_by_label(SEL_MULTISELECT, label, sidebar)
        # Clear existing selections
        clear_btns = widget.locator('[data-testid="stMultiSelectClear"]')
        if clear_btns.count() > 0:
            clear_btns.first.click()
            self.wait_for_rerun()

        # Type and select each value
        input_el = widget.locator("input")
        for val in values:
            input_el.fill(val)
            self.page.wait_for_selector(
                f'[role="listbox"] [role="option"]:has-text("{val}")',
                timeout=WIDGET_TIMEOUT,
            )
            self.page.locator(
                f'[role="listbox"] [role="option"]:has-text("{val}")'
            ).first.click()
        self.wait_for_rerun()

    # ── Slider ───────────────────────────────────────────────────────

    def set_slider(
        self, label: str, value: float, sidebar: bool = False
    ) -> None:
        """Set a slider value using keyboard interaction.

        The thumb is a visually hidden native range input, so min/max/step come
        off the input itself rather than from ARIA attributes.
        """
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        thumb = widget.locator(SEL_SLIDER_INPUT).first
        # The input is clipped to a 1px box, so focus it rather than click it.
        thumb.focus()
        current = float(thumb.input_value() or 0)
        step = float(thumb.get_attribute("step") or 1)

        if step == 0:
            step = 1

        # Calculate arrow key presses needed
        diff = value - current
        presses = abs(int(round(diff / step)))
        key = "ArrowRight" if diff > 0 else "ArrowLeft"

        # Cap at reasonable number to avoid hanging
        presses = min(presses, 500)
        for _ in range(presses):
            self.page.keyboard.press(key)

        self.wait_for_rerun()

    def get_slider_value(
        self, label: str, sidebar: bool = False
    ) -> float:
        """Read the current slider value off its native range input."""
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        return float(
            widget.locator(SEL_SLIDER_INPUT).first.input_value() or 0
        )

    # ── Checkbox ─────────────────────────────────────────────────────

    def check_box(
        self, label: str, checked: bool = True, sidebar: bool = False
    ) -> None:
        """Set a checkbox to checked or unchecked."""
        widget = self._find_widget_by_label(SEL_CHECKBOX, label, sidebar)
        input_el = widget.locator("input[type='checkbox']")
        is_checked = input_el.is_checked()
        if is_checked != checked:
            widget.locator("label").first.click()
            self.wait_for_rerun()

    def is_checked(
        self, label: str, sidebar: bool = False
    ) -> bool:
        """Return whether a checkbox is checked."""
        widget = self._find_widget_by_label(SEL_CHECKBOX, label, sidebar)
        return widget.locator("input[type='checkbox']").is_checked()

    # ── Toggle ───────────────────────────────────────────────────────

    def toggle(
        self, label: str, on: bool = True, sidebar: bool = False
    ) -> None:
        """Set a toggle switch on or off."""
        # Streamlit toggles may use stCheckbox or stToggle data-testid
        # Try stToggle first, fallback to checkbox-style
        scope = self._scope(sidebar)
        widget = scope.locator(
            f'{SEL_TOGGLE}:has-text("{label}"), '
            f'{SEL_CHECKBOX}:has-text("{label}")'
        ).first
        input_el = widget.locator("input[type='checkbox']")
        is_on = input_el.is_checked()
        if is_on != on:
            widget.locator("label").first.click()
            self.wait_for_rerun()

    # ── Button ───────────────────────────────────────────────────────

    def click_button(
        self, label: str, sidebar: bool = False, timeout: int = LONG_OP_TIMEOUT
    ) -> None:
        """Click a button and wait for the rerun to complete."""
        scope = self._scope(sidebar)
        btn = scope.locator(f'button:has-text("{label}")').first
        btn.click()
        self.wait_for_rerun(timeout=timeout)

    def is_button_disabled(
        self, label: str, sidebar: bool = False
    ) -> bool:
        """Check if a button is disabled."""
        scope = self._scope(sidebar)
        btn = scope.locator(f'button:has-text("{label}")').first
        return btn.is_disabled()

    # ── Text Input ───────────────────────────────────────────────────

    def set_text_input(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Clear and type into a text input."""
        widget = self._find_widget_by_label(SEL_TEXT_INPUT, label, sidebar)
        input_el = widget.locator("input").first
        input_el.click()
        input_el.fill(value)
        # Streamlit text inputs commit on blur or Enter
        input_el.press("Enter")
        self.wait_for_rerun()

    def get_text_input_value(
        self, label: str, sidebar: bool = False
    ) -> str:
        """Read current text input value."""
        widget = self._find_widget_by_label(SEL_TEXT_INPUT, label, sidebar)
        return widget.locator("input").first.input_value()

    # ── Number Input ─────────────────────────────────────────────────

    def set_number_input(
        self, label: str, value: float, sidebar: bool = False
    ) -> None:
        """Clear and type into a number input."""
        widget = self._find_widget_by_label(SEL_NUMBER_INPUT, label, sidebar)
        input_el = widget.locator("input").first
        input_el.click()
        input_el.fill(str(value))
        input_el.press("Enter")
        self.wait_for_rerun()

    def get_number_input_value(
        self, label: str, sidebar: bool = False
    ) -> float:
        """Read current number input value."""
        widget = self._find_widget_by_label(SEL_NUMBER_INPUT, label, sidebar)
        val = widget.locator("input").first.input_value()
        try:
            return float(val.replace(",", ""))
        except ValueError:
            return 0.0

    # ── Text Area ────────────────────────────────────────────────────

    def set_text_area(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Clear and type into a text area."""
        widget = self._find_widget_by_label(SEL_TEXT_AREA, label, sidebar)
        textarea = widget.locator("textarea").first
        textarea.click()
        textarea.fill(value)
        # Ctrl+Enter to commit in Streamlit text areas
        textarea.press("Control+Enter")
        self.wait_for_rerun()

    def get_text_area_value(
        self, label: str, sidebar: bool = False
    ) -> str:
        """Read current text area value."""
        widget = self._find_widget_by_label(SEL_TEXT_AREA, label, sidebar)
        return widget.locator("textarea").first.input_value()

    # ── Date Input ───────────────────────────────────────────────────

    def set_date_input(
        self, label: str, date_str: str, sidebar: bool = False
    ) -> None:
        """Set a date input value (format: YYYY/MM/DD or MM/DD/YYYY)."""
        widget = self._find_widget_by_label(SEL_DATE_INPUT, label, sidebar)
        input_el = widget.locator("input").first
        input_el.click()
        # Select all existing text and replace
        self.page.keyboard.press("Control+a")
        self.page.keyboard.type(date_str)
        self.page.keyboard.press("Enter")
        self.wait_for_rerun()

    # ── File Uploader ────────────────────────────────────────────────

    def upload_file(
        self, label: str, file_path: str, sidebar: bool = False
    ) -> None:
        """Upload a file to a file_uploader widget."""
        widget = self._find_widget_by_label(
            SEL_FILE_UPLOADER, label, sidebar
        )
        input_el = widget.locator('input[type="file"]')
        input_el.set_input_files(file_path)
        self.wait_for_rerun()

    # ── Expander ─────────────────────────────────────────────────────

    def expand_expander(self, label: str, sidebar: bool = False) -> None:
        """Click an expander to open it (if closed)."""
        scope = self._scope(sidebar)
        expander = scope.locator(
            f'{SEL_EXPANDER}:has-text("{label}")'
        ).first
        summary = expander.locator("summary, [data-testid='stExpanderToggleDetails']").first
        # Check if already expanded via aria-expanded or details[open]
        details = expander.locator("details")
        if details.count() > 0:
            is_open = details.get_attribute("open") is not None
            if not is_open:
                summary.click()
                self.page.wait_for_timeout(500)
        else:
            summary.click()
            self.page.wait_for_timeout(500)

    def collapse_expander(self, label: str, sidebar: bool = False) -> None:
        """Click an expander to close it (if open)."""
        scope = self._scope(sidebar)
        expander = scope.locator(
            f'{SEL_EXPANDER}:has-text("{label}")'
        ).first
        details = expander.locator("details")
        if details.count() > 0:
            is_open = details.get_attribute("open") is not None
            if is_open:
                summary = expander.locator("summary").first
                summary.click()
                self.page.wait_for_timeout(500)

    # ── Radio ────────────────────────────────────────────────────────

    def select_radio(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Select a radio option by its text."""
        widget = self._find_widget_by_label(SEL_RADIO, label, sidebar)
        option = widget.locator(f'label:has-text("{value}")').first
        option.click()
        self.wait_for_rerun()

    # ── Select Slider ────────────────────────────────────────────────

    def set_select_slider(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Set a select_slider by using keyboard to cycle to the value.

        A select_slider's labels live in ``aria-valuetext``; its range input
        holds only the numeric index.
        """
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        thumb = widget.locator(SEL_SLIDER_INPUT).first
        thumb.focus()

        # Try pressing right until we find the value (max 50 presses)
        for _ in range(50):
            current = thumb.get_attribute("aria-valuetext") or ""
            if value.lower() in current.lower():
                break
            self.page.keyboard.press("ArrowRight")
        else:
            # Try going left from the start
            self.page.keyboard.press("Home")
            for _ in range(50):
                current = thumb.get_attribute("aria-valuetext") or ""
                if value.lower() in current.lower():
                    break
                self.page.keyboard.press("ArrowRight")

        self.wait_for_rerun()

    # ── Symbol Picker (st_searchbox custom component) ────────────────

    def symbol_picker_count(self) -> int:
        """Number of symbol pickers rendered on the current page."""
        return self.page.locator(SEL_SEARCHBOX).count()

    def set_symbol(self, value: str) -> None:
        """Type a symbol into the picker and commit the first suggestion.

        The picker is a custom component in an iframe, and its suggestion list
        renders inside that same iframe, so both have to be reached through a
        frame locator.
        """
        box = self.page.frame_locator(SEL_SEARCHBOX).first
        inp = box.locator("input").first
        inp.click()
        inp.fill(value)
        # Wait for the suggestion list to catch up with the typed text.
        box.locator(SEL_OPTION).first.wait_for(
            state="visible", timeout=WIDGET_TIMEOUT
        )
        inp.press("Enter")
        self.wait_for_rerun()

    def get_symbol_suggestions(self, query: str) -> list[str]:
        """Type *query* and return the suggestion labels, without committing."""
        box = self.page.frame_locator(SEL_SEARCHBOX).first
        inp = box.locator("input").first
        inp.click()
        inp.fill(query)
        try:
            box.locator(SEL_OPTION).first.wait_for(
                state="visible", timeout=WIDGET_TIMEOUT
            )
        except Exception:
            return []
        opts = box.locator(SEL_OPTION)
        return [opts.nth(i).inner_text() for i in range(opts.count())]

    # ── Date Presets ─────────────────────────────────────────────────

    def click_date_preset(self, key_prefix: str, preset: str) -> None:
        """Click a date-range preset button, e.g. ``("bt", "5Y")``."""
        self.page.locator(
            f'[class*="st-key-{key_prefix}_dp_{preset}"] button'
        ).first.click()
        self.wait_for_rerun()

    def get_active_date_preset(self, key_prefix: str) -> str | None:
        """Return the highlighted date preset, which renders as a primary button."""
        for preset in DATE_PRESETS:
            btn = self.page.locator(
                f'[class*="st-key-{key_prefix}_dp_{preset}"] '
                f'[data-testid="stBaseButton-primary"]'
            )
            if btn.count() > 0:
                return preset
        return None

    # ── Assertion Helpers ────────────────────────────────────────────

    def has_error(self, text: str | None = None) -> bool:
        """Check if an st.error message is visible."""
        alerts = self.page.locator(f'{ALERT}[data-baseweb="notification"]')
        if alerts.count() == 0:
            # Fallback: look for error-styled elements
            alerts = self.page.locator('.stAlert [role="alert"]')
        for i in range(alerts.count()):
            alert = alerts.nth(i)
            # Error alerts typically have a red/error icon or specific class
            content = alert.inner_text()
            css_class = alert.get_attribute("class") or ""
            if "error" in css_class.lower() or (
                text and text.lower() in content.lower()
            ):
                return True
        if text:
            return self.page.locator(f'text="{text}"').count() > 0
        return False

    def has_warning(self, text: str | None = None) -> bool:
        """Check if an st.warning message is visible."""
        return self._has_alert_type("warning", text)

    def has_info(self, text: str | None = None) -> bool:
        """Check if an st.info message is visible."""
        return self._has_alert_type("info", text)

    def has_success(self, text: str | None = None) -> bool:
        """Check if an st.success message is visible."""
        return self._has_alert_type("success", text)

    def _has_alert_type(
        self, alert_kind: str, text: str | None = None
    ) -> bool:
        """Generic alert check. Streamlit renders alerts with icon + text."""
        # Streamlit alert selectors vary by version; cast a wide net
        selectors = [
            f'[data-testid="stAlert"] [data-testid="stAlertContent{alert_kind.capitalize()}"]',
            f'[data-testid="stAlert{alert_kind.capitalize()}"]',
            'div[role="alert"]',
        ]
        for sel in selectors:
            elements = self.page.locator(sel)
            for i in range(elements.count()):
                el_text = elements.nth(i).inner_text()
                if text is None:
                    return True
                if text.lower() in el_text.lower():
                    return True

        # Broader fallback: search all visible text
        if text:
            return self.page.locator(f":text('{text}')").count() > 0
        return False

    def _appears(self, selector: str, timeout: int = 10_000) -> bool:
        """True if *selector* is on the page, waiting up to *timeout* for it.

        Dataframes and Plotly charts are heavy JS components that mount a beat
        after the rerun that produced them settles. A bare ``count()`` raced
        them: with the network in the loop the fetch was slow enough to hide it,
        and against an offline server the same assertion started failing on a
        page that renders perfectly.
        """
        try:
            self.page.wait_for_selector(selector, state="attached", timeout=timeout)
            return True
        except Exception:
            return False

    def has_dataframe(self, timeout: int = 10_000) -> bool:
        """Check if at least one dataframe/table is rendered."""
        return self._appears(DATAFRAME, timeout)

    def has_plotly_chart(self, timeout: int = 10_000) -> bool:
        """Check if at least one Plotly chart is rendered."""
        return self._appears(PLOTLY_CHART, timeout)

    def get_metric_count(self) -> int:
        """Count metric elements (st.metric OR custom .metric-card)."""
        st_count = self.page.locator(METRIC).count()
        card_count = self.page.locator(METRIC_CARD).count()
        return st_count + card_count

    def get_metric_values(self) -> dict[str, str]:
        """Return {label: value} for all visible metric elements.

        Supports both st.metric (data-testid) and custom HTML metric
        cards (.metric-card with .metric-card-label / .metric-card-value).
        """
        result = {}
        # Standard st.metric
        metrics = self.page.locator(METRIC)
        for i in range(metrics.count()):
            m = metrics.nth(i)
            label_el = m.locator('[data-testid="stMetricLabel"]')
            value_el = m.locator('[data-testid="stMetricValue"]')
            if label_el.count() > 0 and value_el.count() > 0:
                result[label_el.inner_text().strip()] = (
                    value_el.inner_text().strip()
                )
        # Custom HTML metric cards
        cards = self.page.locator(METRIC_CARD)
        for i in range(cards.count()):
            card = cards.nth(i)
            label_el = card.locator(METRIC_CARD_LABEL)
            value_el = card.locator(METRIC_CARD_VALUE)
            if label_el.count() > 0 and value_el.count() > 0:
                result[label_el.inner_text().strip()] = (
                    value_el.inner_text().strip()
                )
        return result

    def get_visible_text(self) -> str:
        """Return all visible text in the main content area."""
        return self.page.locator(MAIN).inner_text()

    def get_sidebar_text(self) -> str:
        """Return all visible text in the sidebar."""
        return self.page.locator(SIDEBAR).inner_text()

    def screenshot(self, path: str) -> None:
        """Save a screenshot for debugging."""
        self.page.screenshot(path=path)

    # ── Convenience ──────────────────────────────────────────────────

    def widget_exists(
        self, selector: str, label: str, sidebar: bool = False
    ) -> bool:
        """Check if a widget with the given label exists."""
        scope = self._scope(sidebar)
        return (
            scope.locator(f'{selector}:has-text("{label}")').count() > 0
        )

    def count_widgets(
        self, selector: str, sidebar: bool = False
    ) -> int:
        """Count widgets matching a selector."""
        scope = self._scope(sidebar)
        return scope.locator(selector).count()
