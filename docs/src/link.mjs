const plugin = {
  name: 'inlineCode to link',
  transforms: [
    {
      name: 'transform-inline-code',
      doc: 'An example transform that rewrites inline code into ref',
      stage: 'document',
      plugin: (_, utils) => (node) => {
        utils.selectAll('inlineCode', node).forEach((inlineCodeNode) => {
          // const childTextNodes = utils.selectAll('value', inlineCodeNode);
          // const childText = childTextNodes.map((child) => child.value).join('');
          console.log(inlineCodeNode.type, inlineCodeNode.value)
        });
      },
    },
  ],
};

export default plugin;