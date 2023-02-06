import React from 'react';
import ReactDOM from 'react-dom';

import MyVideos from './GamePage';

const videos = [
  {
    id: '1',
    name: 'Video 1',
    url: 'http://example.com/video1',
    thumbnail: 'http://example.com/thumbnail1.jpg',
    description: 'This is the first video.',
  },
  {
    id: '2',
    name: 'Video 2',
    url: 'http://example.com/video2',
    thumbnail: 'http://example.com/thumbnail2.jpg',
    description: 'This is the second video.',
  },
  {
    id: '3',
    name: 'Video 3',
    url: 'http://example.com/video3',
    thumbnail: 'http://example.com/thumbnail3.jpg',
    description: 'This is the third video.',
  },
];

ReactDOM.render(
  <MyVideos videos={videos} />,
  document.getElementById('root')
);