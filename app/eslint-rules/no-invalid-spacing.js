/** @type {import('eslint').Rule.RuleModule} */
module.exports = {
  meta: {
    type: 'suggestion',
    docs: {
      description: 'Enforce spacing values to be multiples of 4 (2 is also allowed)',
    },
    schema: [],
    messages: {
      invalidSpacing:
        'Spacing value {{value}} must be a multiple of 4 (or 2). Got {{value}}.',
    },
  },
  create(context) {
    const SPACING_PROPS = new Set([
      'padding',
      'paddingTop',
      'paddingBottom',
      'paddingLeft',
      'paddingRight',
      'paddingHorizontal',
      'paddingVertical',
      'margin',
      'marginTop',
      'marginBottom',
      'marginLeft',
      'marginRight',
      'marginHorizontal',
      'marginVertical',
      'gap',
      'rowGap',
      'columnGap',
    ]);

    function isValid(value) {
      return value === 0 || value === 2 || value % 4 === 0;
    }

    function checkProperties(properties) {
      for (const prop of properties) {
        if (prop.type !== 'Property') continue;
        const key = prop.key?.name ?? prop.key?.value;
        if (!SPACING_PROPS.has(key)) continue;
        if (prop.value.type === 'Literal' && typeof prop.value.value === 'number') {
          if (!isValid(prop.value.value)) {
            context.report({
              node: prop.value,
              messageId: 'invalidSpacing',
              data: { value: prop.value.value },
            });
          }
        }
      }
    }

    return {
      // StyleSheet.create({ ruleName: { padding: X } })
      CallExpression(node) {
        if (
          node.callee.type !== 'MemberExpression' ||
          node.callee.object.name !== 'StyleSheet' ||
          node.callee.property.name !== 'create' ||
          node.arguments.length === 0
        ) return;

        const stylesObj = node.arguments[0];
        if (stylesObj.type !== 'ObjectExpression') return;

        for (const styleRule of stylesObj.properties) {
          if (
            styleRule.type === 'Property' &&
            styleRule.value.type === 'ObjectExpression'
          ) {
            checkProperties(styleRule.value.properties);
          }
        }
      },
    };
  },
};
