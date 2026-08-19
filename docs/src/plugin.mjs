
const crossLinker = {
  name: "cl",
  doc: "Automatically generate cross links",
  body: { type: 'myst', doc: "The body of the role.", required: true },
  run(data, vfile, ctx) {
    const name = data.body[0].value
    const base = name.toLowerCase().match(/[a-z]*/)[0]
    const url = "#" + base + "_def" 
    return [
        {
        type: 'link',
        url: url,
        children: [
            {
            type: 'text',
            value: name
            }
        ]
        }
    ];
    
  },
};

const plugin = {
  name: "Auto Cross Linker",
  roles: [crossLinker],
};

export default plugin;