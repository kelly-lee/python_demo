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

 Date: 17/01/2019 19:08:24
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for nasdaq_daily
-- ----------------------------
DROP TABLE IF EXISTS `nasdaq_daily`;
CREATE TABLE `nasdaq_daily` (
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
  KEY `ix_nasdaq_daily_Date` (`date`)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
