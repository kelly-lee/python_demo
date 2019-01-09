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

 Date: 09/01/2019 19:02:05
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for cpi
-- ----------------------------
DROP TABLE IF EXISTS `cpi`;
CREATE TABLE `cpi` (
  `index` bigint(20) DEFAULT NULL,
  `month` text,
  `cpi` double DEFAULT NULL,
  KEY `ix_cpi_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for deposit_rate
-- ----------------------------
DROP TABLE IF EXISTS `deposit_rate`;
CREATE TABLE `deposit_rate` (
  `index` bigint(20) DEFAULT NULL,
  `date` text,
  `deposit_type` text,
  `rate` text,
  KEY `ix_deposit_rate_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for gdp_contrib
-- ----------------------------
DROP TABLE IF EXISTS `gdp_contrib`;
CREATE TABLE `gdp_contrib` (
  `index` bigint(20) DEFAULT NULL,
  `year` bigint(20) DEFAULT NULL,
  `gdp_yoy` double DEFAULT NULL,
  `pi` double DEFAULT NULL,
  `si` double DEFAULT NULL,
  `industry` double DEFAULT NULL,
  `ti` double DEFAULT NULL,
  KEY `ix_gdp_contrib_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for gdp_for
-- ----------------------------
DROP TABLE IF EXISTS `gdp_for`;
CREATE TABLE `gdp_for` (
  `index` bigint(20) DEFAULT NULL,
  `year` bigint(20) DEFAULT NULL,
  `end_for` double DEFAULT NULL,
  `for_rate` double DEFAULT NULL,
  `asset_for` double DEFAULT NULL,
  `asset_rate` double DEFAULT NULL,
  `goods_for` double DEFAULT NULL,
  `goods_rate` double DEFAULT NULL,
  KEY `ix_gdp_for_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for gdp_quarter
-- ----------------------------
DROP TABLE IF EXISTS `gdp_quarter`;
CREATE TABLE `gdp_quarter` (
  `index` bigint(20) DEFAULT NULL,
  `quarter` double DEFAULT NULL,
  `gdp` double DEFAULT NULL,
  `gdp_yoy` double DEFAULT NULL,
  `pi` double DEFAULT NULL,
  `pi_yoy` double DEFAULT NULL,
  `si` double DEFAULT NULL,
  `si_yoy` double DEFAULT NULL,
  `ti` double DEFAULT NULL,
  `ti_yoy` double DEFAULT NULL,
  KEY `ix_gdp_quarter_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for gdp_year
-- ----------------------------
DROP TABLE IF EXISTS `gdp_year`;
CREATE TABLE `gdp_year` (
  `index` bigint(20) DEFAULT NULL,
  `year` bigint(20) DEFAULT NULL,
  `gdp` double DEFAULT NULL,
  `pc_gdp` double DEFAULT NULL,
  `gnp` double DEFAULT NULL,
  `pi` double DEFAULT NULL,
  `si` double DEFAULT NULL,
  `industry` double DEFAULT NULL,
  `cons_industry` double DEFAULT NULL,
  `ti` double DEFAULT NULL,
  `trans_industry` double DEFAULT NULL,
  `lbdy` double DEFAULT NULL,
  KEY `ix_gdp_year_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for loan_rate
-- ----------------------------
DROP TABLE IF EXISTS `loan_rate`;
CREATE TABLE `loan_rate` (
  `index` bigint(20) DEFAULT NULL,
  `date` text,
  `loan_type` text,
  `rate` text,
  KEY `ix_loan_rate_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for money_supply
-- ----------------------------
DROP TABLE IF EXISTS `money_supply`;
CREATE TABLE `money_supply` (
  `index` bigint(20) DEFAULT NULL,
  `month` text,
  `m2` text,
  `m2_yoy` text,
  `m1` text,
  `m1_yoy` text,
  `m0` text,
  `m0_yoy` text,
  `cd` text,
  `cd_yoy` text,
  `qm` text,
  `qm_yoy` text,
  `ftd` text,
  `ftd_yoy` text,
  `sd` text,
  `sd_yoy` text,
  `rests` text,
  `rests_yoy` text,
  KEY `ix_money_supply_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for money_supply_bal
-- ----------------------------
DROP TABLE IF EXISTS `money_supply_bal`;
CREATE TABLE `money_supply_bal` (
  `index` bigint(20) DEFAULT NULL,
  `year` text,
  `m2` text,
  `m1` text,
  `m0` text,
  `cd` text,
  `qm` text,
  `ftd` text,
  `sd` text,
  `rests` text,
  KEY `ix_money_supply_bal_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for ppi
-- ----------------------------
DROP TABLE IF EXISTS `ppi`;
CREATE TABLE `ppi` (
  `index` bigint(20) DEFAULT NULL,
  `month` text,
  `ppiip` double DEFAULT NULL,
  `ppi` double DEFAULT NULL,
  `qm` double DEFAULT NULL,
  `rmi` double DEFAULT NULL,
  `pi` double DEFAULT NULL,
  `cg` double DEFAULT NULL,
  `food` double DEFAULT NULL,
  `clothing` double DEFAULT NULL,
  `roeu` double DEFAULT NULL,
  `dcg` double DEFAULT NULL,
  KEY `ix_ppi_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for report_data
-- ----------------------------
DROP TABLE IF EXISTS `report_data`;
CREATE TABLE `report_data` (
  `index` bigint(20) DEFAULT NULL,
  `code` varchar(16) DEFAULT NULL,
  `name` varchar(16) DEFAULT NULL,
  `eps` double DEFAULT NULL,
  `eps_yoy` double DEFAULT NULL,
  `bvps` double DEFAULT NULL,
  `roe` double DEFAULT NULL,
  `epcf` double DEFAULT NULL,
  `net_profits` double DEFAULT NULL,
  `profits_yoy` double DEFAULT NULL,
  `distrib` varchar(128) DEFAULT NULL,
  `report_date` varchar(16) DEFAULT NULL,
  `year` bigint(20) DEFAULT NULL,
  `season` bigint(20) DEFAULT NULL,
  KEY `ix_report_data_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for rrr
-- ----------------------------
DROP TABLE IF EXISTS `rrr`;
CREATE TABLE `rrr` (
  `index` bigint(20) DEFAULT NULL,
  `date` text,
  `before` text,
  `now` text,
  `changed` text,
  KEY `ix_rrr_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------
-- Table structure for stock_basics
-- ----------------------------
DROP TABLE IF EXISTS `stock_basics`;
CREATE TABLE `stock_basics` (
  `code` varchar(16) CHARACTER SET utf8mb4 NOT NULL,
  `name` varchar(16) CHARACTER SET utf8mb4 DEFAULT NULL,
  `industry` varchar(32) CHARACTER SET utf8mb4 DEFAULT NULL,
  `area` varchar(32) CHARACTER SET utf8mb4 DEFAULT NULL,
  `pe` double DEFAULT NULL,
  `outstanding` double DEFAULT NULL,
  `totals` double DEFAULT NULL,
  `totalAssets` double DEFAULT NULL,
  `liquidAssets` double DEFAULT NULL,
  `fixedAssets` double DEFAULT NULL,
  `reserved` double DEFAULT NULL,
  `reservedPerShare` double DEFAULT NULL,
  `esp` double DEFAULT NULL,
  `bvps` double DEFAULT NULL,
  `pb` double DEFAULT NULL,
  `timeToMarket` bigint(20) DEFAULT NULL,
  `undp` double DEFAULT NULL,
  `perundp` double DEFAULT NULL,
  `rev` double DEFAULT NULL,
  `profit` double DEFAULT NULL,
  `gpr` double DEFAULT NULL,
  `npr` double DEFAULT NULL,
  `holders` double DEFAULT NULL,
  PRIMARY KEY (`code`),
  UNIQUE KEY `idx_code` (`code`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

SET FOREIGN_KEY_CHECKS = 1;
