// /js/entities/entity.js

/** Entity Class
 * This module contains the main entity class.
 */
function Entity(groupName, point, speed, health, sprite) {

    var _id = 'e' + Entity.prototype._count++;
    this.id = _id;
    this.group = groupName;
    this.collides = true;

    var x = 0,
        y = 0,
        spriteHeight = 0,
        spriteWidth = 0,
        spriteImage = null,
        entitySpeed = speed || 0;
        entityHealth = health || 1;

    if (point) {
        x = point.x;
        y = point.y;
    }
    if (sprite) {
        spriteHeight = sprite.height;
        spriteWidth = sprite.width;
        spriteImage = sprite.image;
    }

    // Create the initial state
    this.state = {
        killed: false,
        position: {
            x: x,
            y: y
        },
        moveSpeed: entitySpeed,
        health: entityHealth
    };

    this.sprite = {
        height: spriteHeight,
        width: spriteWidth,
        image: spriteImage
    };

    return this;
}

Entity.prototype.constructor = Entity;

Entity.prototype._count = 0;

Entity.prototype.inBoundary = function(scope) {
    var maxHeight = scope.constants.height - this.sprite.height,
        maxWidth = scope.constants.width - this.sprite.width,
        x = this.state.position.x,
        y = this.state.position.y;
    if (x < 0 || x > maxWidth) return false;
    if (y < 0 || y > maxHeight) return false;
    return true;
};

Entity.prototype.haveCollidedWith = function(entity) {
    var h = entity.sprite.height,
        w = entity.sprite.width,
        x = this.state.position.x,
        y = this.state.position.y,
        _x = entity.state.position.x,
        _y = entity.state.position.y;
        
    if (x >= _x && y >= _y) {
        if (x <= _x * w && y <= _y * h) {
            return true;
        }
    } return false;
};

Entity.prototype.kill = function() {
        this.state.killed = true;
        return this;
    };

module.exports = Entity;