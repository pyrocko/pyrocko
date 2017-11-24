function load_markers(fn){
      try{
          return loadXMLDoc(fn);
      }
      catch(err)
      {
          txt="An error occurred while trying to read dumped pyrocko marker.\n\n";
          txt+="Probably, your browser does not allow to open that document\n\n";
          txt+="due to the \"Same-Origin-Policy\".\n\n";
          txt+="A solution might be to change your default browser.\n\n";
          alert(txt);
      }
}


function get_transformed_lonlat(item){
  var lonLat = new OpenLayers.LonLat(item.lon, item.lat);
  lonLat.transform(new OpenLayers.Projection("EPSG:4326"),
                   map.getProjectionObject());
  return lonLat;
}


function get_standard_station_icon(){
  var size = new OpenLayers.Size(15,15);
  var offset = new OpenLayers.Pixel(-(size.w/2), -size.h);
  var icon = new OpenLayers.Icon('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAbUlEQVQ4ja3OOwrAMAwDUN0rHXzzHK1dQnGJ68gfgTYhHsDlAjDILZW52hIBcK9Kx+FUh2WlqLMWpdaVlZaupLR0aaWnSyk9XVjJ6EJKRkcrIzpKGdEdlRmdq8zofpUVnams6DZlh+6j7NC9ygd+w8cw2AIG3AAAAABJRU5ErkJggg==', size, offset);
  return icon}


function Station(lat, lon, nsl, icon_data){
  this.lat = lat;
  this.lon = lon;
  this.nsl = nsl;
  this.icon_data = icon_data || get_standard_station_icon();
}


function Event(lat, lon, time, depth, magnitude, icon_data){
  this.lat = lat;
  this.lon = lon;
  this.time = time || 0;
  this.depth = depth || 0;
  this.magnitude = magnitude || 0;
  this.get_event_description = function(){
      return "Magnitude " + this.magnitude.toFixed(2) + "<br>" + this.time + "<br>depth: " + (this.depth/1000).toFixed(2) + " km"};

  this.get_fill_color = function(depth_min, depth_max){
      if (depth_min==depth_max){
        scale = 0.5;}
      else{
        var scale = (this.depth-depth_min)/(depth_max-depth_min);}
      var r = parseInt(255 * scale);
        var b = parseInt(255 * (1-scale));
        return 'rgba(' + r + ',0,'+ b + ',0.5)';
  }
}
