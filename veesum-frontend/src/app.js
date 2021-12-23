//setting up a basic server for our application
let express = require( 'express' );
let app = express();
let server = require( 'http' ).Server( app );
let io = require( 'socket.io' )( server );
let stream = require( './ws/stream' );
let path = require( 'path' );
let favicon = require( 'serve-favicon' );

//app.use( favicon( path.join( __dirname, 'favicon.ico' ) ) );

//so the app can use static assets
app.use( '/assets', express.static( path.join( __dirname, 'assets' ) ) );

//reout to the index page when a request is made
app.get( '/', ( req, res ) => {
    res.sendFile( __dirname + '/index.html' );
} );

//we are connecting to a custom namespace called */stream*
io.of( '/stream' ).on( 'connection', stream );

server.listen( process.env.PORT || 3000 );
