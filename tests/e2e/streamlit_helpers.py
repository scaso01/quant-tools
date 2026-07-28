"""Streamlit widget interaction helpers for Playwright E2E tests.

All CSS selectors are isolated here so Streamlit version upgrades
only require changes in this one file.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

# ── Selector Constants ───────────────────────────────────────────────

SIDEBAR = 'section[data-testid="stSidebar"]'
MAIN = '[data-testid="stAppViewContainer"]'
STATUS_WIDGET = '[data-testid="stStatusWidget"]'
ALERT = '[data-testid="stAlert"]'
DATAFRAME = '[data-testid="stDataFrame"]'
PLOTLY_CHART = ".stPlotlyChart, .js-plotly-plot"
# st.metric elements (standard) + custom HTML metric cards
METRIC = '[data-testid="stMetric"]'
METRIC_CARD = ".metric-card"
METRIC_CARD_LABEL = ".metric-card-label"
METRIC_CARD_VALUE = ".metric-card-value"

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
        """Click a sidebar radio option to navigate to a page.

        ``page_name`` should be the full display name including emoji,
        e.g. "📡 Market Scanner".
        """
        sidebar = self.page.locator(SIDEBAR)
        # Streamlit radio options are rendered as label elements
        option = sidebar.locator(
            f'{SEL_RADIO} label:has-text("{page_name}")'
        )
        if option.count() == 0:
            # Fallback: match partial text (without emoji)
            clean = page_name.lstrip()
            for ch in page_name:
                if ch.isalpha():
                    break
                clean = page_name[page_name.index(ch) + 1 :].lstrip()
            option = sidebar.locator(
                f'{SEL_RADIO} label:has-text("{clean}")'
            )
        option.first.click()
        self.wait_for_rerun()

    def get_page_names(self) -> list[str]:
        """Return all navigation radio option texts."""
        sidebar = self.page.locator(SIDEBAR)
        labels = sidebar.locator(f"{SEL_RADIO} label")
        return [labels.nth(i).inner_text() for i in range(labels.count())]

    # ── Wait Helpers ─────────────────────────────────────────────────

    def wait_for_rerun(self, timeout: int = RERUN_TIMEOUT) -> None:
        """Wait for Streamlit's script rerun to complete.

        Streamlit shows a status widget while re-running. This waits
        for it to appear (briefly) and then disappear.
        """
        try:
            # Wait for the running indicator to appear (short timeout —
            # fast reruns may never show it)
            self.page.wait_for_selector(
                STATUS_WIDGET, state="visible", timeout=1_000
            )
        except Exception:
            pass  # May already be done

        # Now wait for it to vanish
        try:
            self.page.wait_for_selector(
                STATUS_WIDGET, state="hidden", timeout=timeout
            )
        except Exception:
            pass  # May never have appeared

        # Extra safety: wait for network idle
        self.page.wait_for_load_state("networkidle", timeout=5_000)

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

    def select_box(
        self, label: str, value: str, sidebar: bool = False
    ) -> None:
        """Open a Streamlit selectbox and choose a value."""
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        # Click the current value to open dropdown
        widget.locator('[data-baseweb="select"]').click()
        # Wait for dropdown list to appear
        self.page.wait_for_selector(
            '[data-baseweb="popover"] li, [role="listbox"] [role="option"]',
            timeout=WIDGET_TIMEOUT,
        )
        # Click the desired option
        option = self.page.locator(
            f'[data-baseweb="popover"] li:has-text("{value}"), '
            f'[role="listbox"] [role="option"]:has-text("{value}")'
        ).first
        option.click()
        self.wait_for_rerun()

    def get_selectbox_value(
        self, label: str, sidebar: bool = False
    ) -> str:
        """Read the current value of a selectbox."""
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        return widget.locator('[data-baseweb="select"]').inner_text().strip()

    def get_selectbox_options(
        self, label: str, sidebar: bool = False
    ) -> list[str]:
        """Open a selectbox and return all option texts, then close it."""
        widget = self._find_widget_by_label(SEL_SELECTBOX, label, sidebar)
        widget.locator('[data-baseweb="select"]').click()
        self.page.wait_for_selector(
            '[data-baseweb="popover"] li, [role="listbox"] [role="option"]',
            timeout=WIDGET_TIMEOUT,
        )
        options_loc = self.page.locator(
            '[data-baseweb="popover"] li, [role="listbox"] [role="option"]'
        )
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
        """Set a slider value using keyboard interaction."""
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        thumb = widget.locator('[role="slider"]').first
        thumb.click()
        # Read current value from aria-valuenow
        current = float(thumb.get_attribute("aria-valuenow") or 0)
        min_val = float(thumb.get_attribute("aria-valuemin") or 0)
        max_val = float(thumb.get_attribute("aria-valuemax") or 100)
        step = float(thumb.get_attribute("aria-valuestep") or 1)

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
        """Read current slider value from aria-valuenow."""
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        thumb = widget.locator('[role="slider"]').first
        return float(thumb.get_attribute("aria-valuenow") or 0)

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
        """Set a select_slider by using keyboard to cycle to the value."""
        widget = self._find_widget_by_label(SEL_SLIDER, label, sidebar)
        thumb = widget.locator('[role="slider"]').first
        thumb.click()

        # Try pressing right until we find the value (max 50 presses)
        for _ in range(50):
            current = thumb.get_attribute("aria-valuenow") or ""
            if value.lower() in current.lower():
                break
            self.page.keyboard.press("ArrowRight")
        else:
            # Try going left from the start
            self.page.keyboard.press("Home")
            for _ in range(50):
                current = thumb.get_attribute("aria-valuenow") or ""
                if value.lower() in current.lower():
                    break
                self.page.keyboard.press("ArrowRight")

        self.wait_for_rerun()

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
            f'div[role="alert"]',
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

    def has_dataframe(self) -> bool:
        """Check if at least one dataframe/table is rendered."""
        return self.page.locator(DATAFRAME).count() > 0

    def has_plotly_chart(self) -> bool:
        """Check if at least one Plotly chart is rendered."""
        return self.page.locator(PLOTLY_CHART).count() > 0

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
