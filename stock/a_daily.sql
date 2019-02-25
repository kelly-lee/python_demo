/*
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 50637
 Source Host           : localhost:3306
 Source Schema         : Stock

 Target Server Type    : MySQL
 Target Server Version : 50637
 File Encoding         : 65001

 Date: 25/02/2019 19:23:48
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for a_daily
-- ----------------------------
DROP TABLE IF EXISTS `a_daily`;
CREATE TABLE `a_daily` (
  `date` date NOT NULL,
  `high` double NOT NULL,
  `low` double NOT NULL,
  `open` double NOT NULL,
  `close` double NOT NULL,
  `volume` double NOT NULL,
  `adj_close` double NOT NULL,
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `symbol` varchar(16) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_date` (`date`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=214246 DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
