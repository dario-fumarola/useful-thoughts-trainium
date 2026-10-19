"""Explicit versioned cohort identities; never change a legacy denominator."""

SCHEMAS = {
    'notes': ('real-trace-prefix-only-notes-v10', 'real-target-prefix-only-notes-v15'),
    'scalar': ('real-trace-native-scalar-secondary-v10', 'real-target-native-scalar-v15'),
}
LEGACY_PANELS = ('DEV_AST256', 'DEV_legacy256')
FRESH_PANEL = 'DEV_fresh384'


def panel_scope(panel, kind):
    if kind not in SCHEMAS:
        raise ValueError('Explicit note or scalar protocol kind required')
    if panel in LEGACY_PANELS:
        return SCHEMAS[kind][0], 256
    if panel == FRESH_PANEL:
        return SCHEMAS[kind][1], 384
    raise ValueError('Declared arithmetic development panel required; never TEST')


def protocol_items(protocol, kind):
    old, new = SCHEMAS[kind]
    schema, items, panel = protocol.get('schema'), protocol.get('items'), protocol.get('panel')
    if schema == old and items == 256 and panel in (*LEGACY_PANELS, None):
        return 256
    if schema == new and items == 384 and panel == FRESH_PANEL:
        return 384
    raise ValueError('Cohort size/panel must match its versioned decoding identity')


def binding_statuses(panel):
    old = 'FROZEN_AUDITED_REAL_TRACE_TERMINAL320'
    return (old, 'FROZEN_AUDITED_REAL_TARGET_TERMINAL320') if panel == FRESH_PANEL else (old,)
