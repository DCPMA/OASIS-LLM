"""Datasets page: list, generate, review, approve curated image sets."""
from __future__ import annotations

from collections import Counter

import streamlit as st

from oasis_llm import datasets as ds
from oasis_llm.dashboard_pages._ui import (
    connect_ro, connect_rw, db_locked_warning, kpi, page_header, status_pill,
)
from oasis_llm.images import image_categories, image_path


def render():
    page_header("Datasets", "Curated image sets — generate, review, approve.", icon="📁")

    detail_id = st.query_params.get("dataset")
    if detail_id:
        _render_detail(detail_id)
        return

    _render_list()


def _render_list():
    con_ro = connect_ro()
    if con_ro is None:
        st.warning("No database yet. Use the form below to generate your first dataset.")
        rows = []
    else:
        rows = ds.list_all(con_ro)

    # KPI strip
    n_total = len(rows)
    n_approved = sum(1 for d in rows if d.status == "approved")
    n_active_total = sum(d.active_count for d in rows)
    _ = n_active_total  # keep for kpi
    cols = st.columns(3)
    cols[0].markdown(kpi("Total datasets", n_total), unsafe_allow_html=True)
    cols[1].markdown(kpi("Approved", n_approved), unsafe_allow_html=True)
    cols[2].markdown(kpi("Active images (sum)", f"{n_active_total:,}"), unsafe_allow_html=True)

    st.markdown("---")

    tab_list, tab_new = st.tabs(["📋  All datasets", "✨  Generate new"])

    with tab_list:
        if not rows:
            st.info("No datasets yet.")
        else:
            # Compact card list
            for d in rows:
                _row_card(d)

    with tab_new:
        _generate_form()


def _row_card(d):
    """Render one dataset row as a clickable card."""
    cols = st.columns([4, 2, 2, 2, 2])
    with cols[0]:
        st.markdown(
            f"<div style='font-weight:600; font-size:1.05rem'>{d.dataset_id}</div>"
            f"<div style='color:#8a8aa0; font-size:0.82rem'>{d.description or ''}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"{status_pill(d.status)} &nbsp; {status_pill(d.source)}",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<div style='color:#8a8aa0; font-size:0.78rem;'>active</div>"
            f"<div style='font-weight:600;'>{d.active_count} / {d.image_count}</div>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"<div style='color:#8a8aa0; font-size:0.78rem;'>created</div>"
            f"<div style='font-size:0.85rem;'>{str(d.created_at)[:19] if d.created_at else '-'}</div>",
            unsafe_allow_html=True,
        )
    with cols[4]:
        if st.button("Open ›", key=f"open_{d.dataset_id}", width='stretch'):
            st.query_params["dataset"] = d.dataset_id
            st.rerun()
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)


def _generate_form():
    import random as _rand
    from datetime import date
    from oasis_llm.images import category_count
    n_cats = max(1, category_count())
    today = date.today().strftime("%Y%m%d")
    st.markdown("Create a new draft dataset by sampling from the OASIS pool.")
    st.caption(
        f"OASIS has **{n_cats} categories**. "
        f"Use *uniform* to get an equal count per category — `n` must be a multiple of {n_cats}."
    )
    strategies = ["uniform", "stratified", "random", "all"]
    with st.form("generate_dataset", clear_on_submit=False):
        c1, c2, c3 = st.columns([3, 2, 2])
        name = c1.text_input("Name", value=today, help="A short slug. Defaults to today.")
        n = c2.number_input("Image count", min_value=1, max_value=900, value=40, step=n_cats)
        strategy = c3.selectbox(
            "Sampling strategy", strategies,
            help=(
                f"uniform = equal per category (n must be ÷{n_cats}) · "
                "stratified = proportional across categories · "
                "random = ignore categories · all = every image"
            ),
        )
        sc1, sc2 = st.columns([1, 3])
        override_seed = sc1.checkbox("Override seed", value=False,
                                     help="Off = random seed each time.")
        seed_val = sc2.number_input("Seed", min_value=0, value=42, disabled=not override_seed)
        description = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Generate draft dataset", type="primary")
    if submitted:
        if strategy == "uniform" and int(n) % n_cats != 0:
            st.error(f"For uniform, n must be a multiple of {n_cats}. Got {n}.")
            return
        seed = int(seed_val) if override_seed else _rand.randint(0, 2**31 - 1)
        con = connect_rw()
        if con is None:
            db_locked_warning()
            return
        try:
            ds_id = ds.generate(
                con, name, int(n), strategy=strategy, seed=seed,
                description=description or None,
            )
        except Exception as e:
            st.error(f"Failed: {e}")
            return
        st.success(f"Created `{ds_id}` with {n} images (seed={seed}). Review & approve below.")
        st.query_params["dataset"] = ds_id
        st.rerun()


def _render_detail(dataset_id: str):
    con_ro = connect_ro()
    if con_ro is None:
        st.error("Database missing.")
        return
    d = ds.get(con_ro, dataset_id)
    if d is None:
        st.error(f"Unknown dataset `{dataset_id}`.")
        if st.button("← Back to list"):
            del st.query_params["dataset"]
            st.rerun()
        return

    # Header
    top = st.columns([6, 2])
    with top[0]:
        st.markdown(
            f"# 📁 {d.dataset_id} &nbsp; {status_pill(d.status)} {status_pill(d.source)}",
            unsafe_allow_html=True,
        )
        if d.description:
            st.caption(d.description)
    with top[1]:
        if st.button("← All datasets", width='stretch'):
            del st.query_params["dataset"]
            st.rerun()

    images = ds.images(con_ro, dataset_id)
    cats = image_categories()
    active_imgs = [r for r in images if not r["excluded"]]
    excluded_imgs = [r for r in images if r["excluded"]]

    # KPIs
    cols = st.columns(4)
    cols[0].markdown(kpi("Total", len(images)), unsafe_allow_html=True)
    cols[1].markdown(kpi("Active", len(active_imgs)), unsafe_allow_html=True)
    cols[2].markdown(kpi("Excluded", len(excluded_imgs)), unsafe_allow_html=True)
    cols[3].markdown(
        kpi("Created", str(d.created_at)[:10] if d.created_at else "—"),
        unsafe_allow_html=True,
    )

    # Action buttons
    is_locked = d.status != "draft" or d.source == "builtin"
    if d.source == "builtin":
        st.info("🔒 Built-in datasets are read-only.")
    elif d.status == "approved":
        st.info(f"🔒 This dataset is **approved** ({str(d.approved_at)[:19]}). Duplicate to edit.")
    elif d.status == "archived":
        st.info("🔒 This dataset is archived.")

    btn_cols = st.columns(6)
    with btn_cols[0]:
        if not is_locked and st.button("✅ Approve", width='stretch', disabled=len(active_imgs) == 0):
            con = connect_rw()
            if con is None:
                db_locked_warning(); return
            try:
                ds.approve(con, dataset_id)
                st.success("Approved!")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with btn_cols[1]:
        if not is_locked and st.button(
            "🎲 Shuffle", width='stretch',
            help="Re-roll the entire dataset using the same generation params and a new random seed.",
        ):
            st.session_state[f"confirm_shuf_{dataset_id}"] = True
    with btn_cols[2]:
        if d.status != "archived" and d.source != "builtin" and st.button("🗄️ Archive", width='stretch'):
            con = connect_rw()
            if con is None: db_locked_warning(); return
            ds.archive(con, dataset_id); st.rerun()
    with btn_cols[3]:
        if st.button("📋 Duplicate", width='stretch'):
            con = connect_rw()
            if con is None: db_locked_warning(); return
            new_id = ds.duplicate(con, dataset_id, new_name=f"{d.dataset_id}-copy")
            st.query_params["dataset"] = new_id
            st.rerun()
    with btn_cols[4]:
        if d.source != "builtin" and st.button("🗑️ Delete", width='stretch', type="secondary"):
            st.session_state[f"confirm_del_{dataset_id}"] = True
    with btn_cols[5]:
        if st.session_state.get(f"confirm_del_{dataset_id}"):
            if st.button("⚠️ Confirm delete", width='stretch', type="primary"):
                con = connect_rw()
                if con is None: db_locked_warning(); return
                ds.delete(con, dataset_id)
                del st.query_params["dataset"]
                st.session_state.pop(f"confirm_del_{dataset_id}", None)
                st.rerun()

    # Shuffle confirmation banner — destructive (replaces all images), so we
    # require an explicit second click.
    if st.session_state.get(f"confirm_shuf_{dataset_id}"):
        sb1, sb2, sb3 = st.columns([5, 1, 1])
        sb1.warning(
            "🎲 Shuffle will **replace every image** in this dataset with a "
            "fresh sample using the stored generation params and a new random "
            "seed. Manually-added images and exclusions will be lost."
        )
        if sb2.button("Confirm shuffle", type="primary", key=f"do_shuf_{dataset_id}"):
            con = connect_rw()
            if con is None: db_locked_warning(); return
            try:
                n_new = ds.shuffle(con, dataset_id)
                st.session_state.pop(f"confirm_shuf_{dataset_id}", None)
                st.success(f"Shuffled — {n_new} fresh images.")
                st.rerun()
            except Exception as e:
                st.error(f"Shuffle failed: {e}")
        if sb3.button("Cancel", key=f"cancel_shuf_{dataset_id}"):
            st.session_state.pop(f"confirm_shuf_{dataset_id}", None); st.rerun()

    st.markdown("---")

    tab_overview, tab_images, tab_add = st.tabs(
        ["📊  Overview", "🖼️  Images", "➕  Add images"]
    )

    with tab_overview:
        # Category breakdown
        col_l, col_r = st.columns([2, 3])
        with col_l:
            st.markdown("### Photos per category")
            cat_counts = Counter(cats.get(r["image_id"], "Unknown") for r in active_imgs)
            if cat_counts:
                cat_data = [
                    {"Category": c, "Active": n,
                     "Pct": f"{100*n/len(active_imgs):.1f}%"}
                    for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])
                ]
                st.dataframe(cat_data, width='stretch', hide_index=True)
            else:
                st.caption("No active images.")
        with col_r:
            st.markdown("### Generation params")
            if d.generation_params:
                st.json(d.generation_params)
            else:
                st.caption("No generation params recorded.")

    with tab_images:
        _render_image_grid(images, cats, dataset_id, is_locked)

    with tab_add:
        _render_add_images(images, cats, dataset_id, is_locked)


def _render_image_grid(images, cats, dataset_id: str, is_locked: bool):
    has_excluded = any(r["excluded"] for r in images)
    show_excluded = st.toggle("Show excluded (legacy)", value=False) if has_excluded else False
    base = images if show_excluded else [r for r in images if not r["excluded"]]

    # Filter controls — category multi-select + image_id substring search.
    fcols = st.columns([3, 4, 2])
    cat_options = sorted({cats.get(r["image_id"], "Unknown") for r in base})
    chosen_cats = fcols[0].multiselect(
        "Filter by category",
        cat_options,
        key=f"grid_cats_{dataset_id}",
        help="Empty = all categories.",
    )
    search = fcols[1].text_input(
        "Search image_id",
        value="",
        key=f"grid_search_{dataset_id}",
        help="Substring match, case-insensitive.",
    )
    n_cols = fcols[2].selectbox(
        "Columns", [3, 4, 5, 6, 8],
        index=2, key=f"grid_ncols_{dataset_id}",
    )

    def _match(row) -> bool:
        if chosen_cats and cats.get(row["image_id"], "Unknown") not in chosen_cats:
            return False
        if search and search.lower() not in row["image_id"].lower():
            return False
        return True

    filtered = [r for r in base if _match(r)]
    n_filtered = sum(1 for r in base if not _match(r))
    st.caption(
        f"{len(filtered)} of {len(base)} shown"
        + (f" ({n_filtered} hidden by filter)" if n_filtered else "")
        + ". " + ("Use 🔁/🎲 to swap an image you don't like." if not is_locked else "")
    )

    # Persistent swap-feedback banner — survives the rerun so the user can
    # see what changed (toast was vanishing too fast).
    swap_key = f"_last_swap_{dataset_id}"
    last = st.session_state.get(swap_key)
    if last:
        old_p, new_p = last["old"], last["new"]
        cb1, cb2 = st.columns([8, 1])
        with cb1:
            st.success(
                f"✅ Swapped **{old_p}** → **{new_p}** "
                f"({last['mode']}, cat={last.get('cat', '?')})"
            )
        if cb2.button("✕", key=f"clr_swap_{dataset_id}", help="Dismiss"):
            del st.session_state[swap_key]
            st.rerun()

    n_cols = int(n_cols)
    rows = [filtered[i:i + n_cols] for i in range(0, len(filtered), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, img in zip(cols, row):
            with col:
                # Robust file lookup — `image_path` raises if the file is
                # missing, so we cannot rely on `.exists()` after calling it.
                # Build the path defensively.
                from oasis_llm.images import IMAGES_DIR
                from pathlib import Path
                p = IMAGES_DIR / f"{img['image_id']}.jpg"
                if isinstance(p, Path) and p.exists():
                    st.image(str(p), width='stretch')
                else:
                    st.caption(f"⚠️ `{img['image_id']}.jpg` missing on disk")
                cat = cats.get(img["image_id"], "?")
                badge_color = "#3a1a1a" if img["excluded"] else "#1a3a25"
                badge_text = "EXCLUDED" if img["excluded"] else "active"
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#8a8aa0;'>{img['image_id']} · {cat}</div>"
                    f"<div style='display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; "
                    f"background:{badge_color}; font-size:0.7rem; margin: 0.15rem 0;'>{badge_text}</div>",
                    unsafe_allow_html=True,
                )
                if not is_locked:
                    iid = img["image_id"]
                    bcols = st.columns(3)
                    if bcols[0].button(
                        "🔁 Same", key=f"regen_same_{dataset_id}_{iid}",
                        help="Replace with another image from the same category",
                        width='stretch',
                    ):
                        con = connect_rw()
                        if con is None:
                            db_locked_warning()
                        else:
                            try:
                                new_id = ds.regenerate_image(
                                    con, dataset_id, iid, same_category=True
                                )
                                st.session_state[swap_key] = {
                                    "old": iid, "new": new_id,
                                    "mode": "same-category",
                                    "cat": cats.get(iid, "?"),
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"Swap failed: {e}")
                    if bcols[1].button(
                        "🎲 Any", key=f"regen_any_{dataset_id}_{iid}",
                        help="Replace with any unused image (ignores category)",
                        width='stretch',
                    ):
                        con = connect_rw()
                        if con is None:
                            db_locked_warning()
                        else:
                            try:
                                new_id = ds.regenerate_image(
                                    con, dataset_id, iid, same_category=False
                                )
                                st.session_state[swap_key] = {
                                    "old": iid, "new": new_id,
                                    "mode": "any-category",
                                    "cat": cats.get(iid, "?"),
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"Swap failed: {e}")
                    if bcols[2].button(
                        "🗑️", key=f"rm_{dataset_id}_{iid}",
                        help="Remove this image from the dataset (no replacement)",
                        width='stretch',
                    ):
                        con = connect_rw()
                        if con is None:
                            db_locked_warning()
                        else:
                            try:
                                ds.remove_image(con, dataset_id, iid)
                                st.session_state[swap_key] = {
                                    "old": iid, "new": "—",
                                    "mode": "removed",
                                    "cat": cats.get(iid, "?"),
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"Remove failed: {e}")


def _render_add_images(images, cats, dataset_id: str, is_locked: bool):
    """Image picker — filter by category, search by id, multi-select, add."""
    if is_locked:
        st.info("This dataset is locked. Duplicate it to add specific images.")
        return

    in_set = {r["image_id"] for r in images}
    pool = sorted(set(cats.keys()) | set(_all_ids()))  # union: csv + on-disk
    available = [iid for iid in pool if iid not in in_set]

    st.markdown(f"**{len(available)}** images available to add ({len(in_set)} already in set).")

    fcols = st.columns([2, 3, 2])
    cat_options = sorted({cats.get(i, "Unknown") for i in pool})
    chosen_cats = fcols[0].multiselect(
        "Filter by category", cat_options,
        help="Empty = all categories.",
    )
    search = fcols[1].text_input(
        "Search image_id", value="",
        help="Substring match, case-insensitive.",
    )
    page_size = fcols[2].selectbox("Page size", [12, 24, 48, 96], index=1)

    def _match(iid: str) -> bool:
        if chosen_cats and cats.get(iid, "Unknown") not in chosen_cats:
            return False
        if search and search.lower() not in iid.lower():
            return False
        return True

    matches = [iid for iid in available if _match(iid)]
    st.caption(f"{len(matches)} match the filters.")

    page_key = f"_add_page_{dataset_id}"
    page = st.session_state.get(page_key, 0)
    total_pages = max(1, (len(matches) + page_size - 1) // page_size)
    page = min(page, total_pages - 1)
    start = page * page_size
    visible = matches[start:start + page_size]

    pcols = st.columns([1, 6, 1])
    if pcols[0].button("◀ Prev", disabled=page == 0, key=f"prev_{dataset_id}"):
        st.session_state[page_key] = max(0, page - 1); st.rerun()
    pcols[1].markdown(
        f"<div style='text-align:center; color:#8a8aa0;'>"
        f"page {page + 1} / {total_pages} — showing {len(visible)} of {len(matches)}"
        f"</div>", unsafe_allow_html=True,
    )
    if pcols[2].button("Next ▶", disabled=page >= total_pages - 1, key=f"next_{dataset_id}"):
        st.session_state[page_key] = page + 1; st.rerun()

    sel_key = f"_add_sel_{dataset_id}"
    selected: set[str] = set(st.session_state.get(sel_key, set()))

    n_cols = 4
    rows = [visible[i:i + n_cols] for i in range(0, len(visible), n_cols)]
    from oasis_llm.images import IMAGES_DIR
    from pathlib import Path
    for row in rows:
        cols = st.columns(n_cols)
        for col, iid in zip(cols, row):
            with col:
                p = IMAGES_DIR / f"{iid}.jpg"
                if p.exists():
                    st.image(str(p), width='stretch')
                else:
                    st.caption(f"⚠️ `{iid}.jpg` missing")
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#8a8aa0;'>"
                    f"{iid} · {cats.get(iid, '?')}"
                    f"</div>", unsafe_allow_html=True,
                )
                checked = iid in selected
                if st.checkbox("Add", value=checked, key=f"chk_{dataset_id}_{iid}"):
                    selected.add(iid)
                else:
                    selected.discard(iid)
    st.session_state[sel_key] = selected

    acols = st.columns([1, 1, 6])
    if acols[0].button(
        f"➕ Add {len(selected)} image(s)",
        type="primary", disabled=not selected,
        key=f"add_btn_{dataset_id}",
    ):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            added, skipped = ds.add_images(con, dataset_id, sorted(selected))
            st.session_state.pop(sel_key, None)
            st.success(f"Added {added}, skipped {skipped}.")
            st.rerun()
        except Exception as e:
            st.error(f"Add failed: {e}")
    if acols[1].button("Clear selection", key=f"clr_sel_{dataset_id}", disabled=not selected):
        st.session_state.pop(sel_key, None); st.rerun()


def _all_ids() -> list[str]:
    """Cached list of all image_ids on disk."""
    from oasis_llm.images import all_image_ids as _aii
    return _aii()
