"""
Trade Playbook Blueprint
Routes: /playbook  (main page)
        /api/playbook/trades          GET/POST
        /api/playbook/trades/<id>     PUT / DELETE
        /api/playbook/price/<symbol>  GET
"""
import os
import sqlite3
from datetime import datetime, date

import yfinance as yf
from flask import Blueprint, render_template, jsonify, request

playbook_bp = Blueprint("playbook", __name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_UPLOAD       = os.path.join(_PROJECT_ROOT, 'uploads', 'playbook')
_DB           = os.path.join(_UPLOAD, 'trades.db')
os.makedirs(_UPLOAD, exist_ok=True)


# ── Database ─────────────────────────────────────────────────────────────────
def _conn():
    return sqlite3.connect(_DB, check_same_thread=False)


def _init():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date_open    TEXT    NOT NULL,
                symbol       TEXT    NOT NULL,
                market       TEXT    NOT NULL DEFAULT 'US',
                entry_price  REAL    NOT NULL,
                stop_price   REAL    NOT NULL,
                target1      REAL,
                target2      REAL,
                qty          REAL    NOT NULL,
                risk_amount  REAL,
                setup_notes  TEXT,
                status       TEXT    NOT NULL DEFAULT 'OPEN',
                date_closed  TEXT,
                exit_price   REAL,
                pnl          REAL,
                exit_notes   TEXT
            )
        """)


_init()


def _row_to_dict(row, cursor):
    return {k[0]: v for k, v in zip(cursor.description, row)}


# ── Pages ─────────────────────────────────────────────────────────────────────
@playbook_bp.route("/playbook")
def playbook_view():
    return render_template("trade_playbook.html")


@playbook_bp.route("/playbook-card")
def playbook_card():
    return render_template("trade_playbook_card.html")


# ── API: Trades ───────────────────────────────────────────────────────────────
@playbook_bp.route("/api/playbook/trades", methods=["GET"])
def get_trades():
    status = request.args.get("status", "ALL")
    with _conn() as c:
        cur = c.execute(
            "SELECT * FROM trades ORDER BY id DESC"
            if status == "ALL" else
            "SELECT * FROM trades WHERE status=? ORDER BY id DESC",
            () if status == "ALL" else (status,),
        )
        rows = [_row_to_dict(r, cur) for r in cur.fetchall()]
    return jsonify(rows)


@playbook_bp.route("/api/playbook/trades", methods=["POST"])
def add_trade():
    d = request.get_json(force=True)
    required = ("symbol", "entry_price", "stop_price", "qty")
    if not all(d.get(k) for k in required):
        return jsonify({"error": "Missing required fields"}), 400
    entry = float(d["entry_price"])
    stop  = float(d["stop_price"])
    qty   = float(d["qty"])
    if entry <= stop:
        return jsonify({"error": "entry_price must be above stop_price"}), 400
    sym = d["symbol"].strip().upper()
    mkt = d.get("market", "US")
    if mkt == "India" and not sym.endswith(".NS"):
        sym += ".NS"
    risk = round((entry - stop) * qty, 2)
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO trades
            (date_open,symbol,market,entry_price,stop_price,target1,target2,qty,risk_amount,setup_notes,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,'OPEN')
        """, (
            d.get("date_open", date.today().isoformat()),
            sym, mkt, entry, stop,
            float(d["target1"]) if d.get("target1") else None,
            float(d["target2"]) if d.get("target2") else None,
            qty, risk,
            d.get("setup_notes", ""),
        ))
        new_id = cur.lastrowid
    return jsonify({"id": new_id, "risk_amount": risk}), 201


@playbook_bp.route("/api/playbook/trades/<int:trade_id>", methods=["PUT"])
def update_trade(trade_id):
    d = request.get_json(force=True)
    exit_price = float(d.get("exit_price", 0))
    status     = d.get("status", "CLOSED")
    notes      = d.get("exit_notes", "")
    with _conn() as c:
        row = c.execute(
            "SELECT entry_price, qty FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Trade not found"}), 404
        pnl = round((exit_price - row[0]) * row[1], 2)
        c.execute("""
            UPDATE trades
            SET status=?, date_closed=?, exit_price=?, pnl=?, exit_notes=?
            WHERE id=?
        """, (status, date.today().isoformat(), exit_price, pnl, notes, trade_id))
    return jsonify({"pnl": pnl})


@playbook_bp.route("/api/playbook/trades/<int:trade_id>", methods=["DELETE"])
def delete_trade(trade_id):
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE id=?", (trade_id,))
    return jsonify({"deleted": trade_id})


# ── API: Live price ───────────────────────────────────────────────────────────
@playbook_bp.route("/api/playbook/price/<path:symbol>")
def live_price(symbol):
    try:
        h = yf.Ticker(symbol.upper()).history(period="2d")
        if not h.empty:
            price = round(float(h["Close"].iloc[-1]), 2)
            prev  = round(float(h["Close"].iloc[-2]), 2) if len(h) > 1 else price
            chg   = round(price - prev, 4)
            chgp  = round(chg / prev * 100 if prev else 0, 2)
            return jsonify({"price": price, "chg": chg, "chgp": chgp})
    except Exception as e:
        pass
    return jsonify({"error": "Price unavailable"}), 503
