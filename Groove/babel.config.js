// module.exports = function(api) {
//   api.cache(true);
//   return {
//     presets: ['babel-preset-expo', ],
//     plugins: ['nativewind/babel', 'expo-router/babel',],
//   };
// };

// module.exports = function(api) {
//   api.cache(true);
//   return {
//     presets: [
//             ['babel-preset-expo', { jsxImportSource: 'nativewind' }],
//             'nativewind/babel',
//         ],
//     plugins: [
//       'react-native-reanimated/plugin',
//     ],
//   };
// }; 

module.exports = function (api) {
  api.cache(true)
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      [
        '@tamagui/babel-plugin',
        {
          components: ['tamagui'],
          config: './tamagui.config.ts',
          logTimings: true,
          disableExtraction: process.env.NODE_ENV === 'development',
        },
      ],

      // NOTE: this is only necessary if you are using reanimated for animations
      'react-native-reanimated/plugin',
    ],
  }
}