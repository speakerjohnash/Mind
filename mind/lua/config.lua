--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
config.alphabet = alphabet