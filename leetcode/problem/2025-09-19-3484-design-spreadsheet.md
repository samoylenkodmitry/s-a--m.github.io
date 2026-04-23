---
layout: leetcode-entry
title: "3484. Design Spreadsheet"
permalink: "/leetcode/problem/2025-09-19-3484-design-spreadsheet/"
leetcode_ui: true
entry_slug: "2025-09-19-3484-design-spreadsheet"
---

[3484. Design Spreadsheet](https://leetcode.com/problems/design-spreadsheet/description/) medium
[blog post](https://leetcode.com/problems/design-spreadsheet/solutions/7204983/kotlin-rust-by-samoylenkodmitry-7wgc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19092025-3484-design-spreadsheet?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GcF4FwNYORw)

![1.webp](/assets/leetcode_daily_images/5ac7fd9d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1117

#### Problem TLDR

Design Spreadsheet: setCell, resetCell, getValue(a+b) #medium #ds

#### Intuition

Rows count is small 1000, we can store all in two-dimensional array. Formula are just a single shot, without going recursive.

#### Approach

* using a HashMap saves LOC and string parsing
* single array: `key = (c[0]-'A') * rows + c[1..].toInt()`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 205ms
class Spreadsheet(rows: Int) : HashMap<String, Int>() {
    fun setCell(c: String, v: Int) = put(c, v)
    fun resetCell(c: String) = put(c, 0)
    fun getValue(f: String) = f.drop(1).split("+")
        .sumOf { if (it[0].isDigit()) it.toInt() else get(it) ?: 0 }
}

```

// 30ms
struct Spreadsheet([i32;26001]); impl Spreadsheet {
    fn new(_: i32) -> Self { Spreadsheet([0;26001]) }
    fn k(&self, c: &str) -> usize { let b = c.as_bytes();
        (b[0]-b'A')as usize*1000+c[1..].parse::<usize>().unwrap()}
    fn set_cell(&mut self, c: String, v: i32) { self.0[self.k(&c.as_str())] = v }
    fn reset_cell(&mut self, c: String) { self.set_cell(c, 0) }
    fn get_value(&self, f: String) -> i32 {
        f[1..].split('+').map(|t|t.parse().unwrap_or_else(|_| self.0[self.k(&t)])).sum()
    }
}

```

