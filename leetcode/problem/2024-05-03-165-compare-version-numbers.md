---
layout: leetcode-entry
title: "165. Compare Version Numbers"
permalink: "/leetcode/problem/2024-05-03-165-compare-version-numbers/"
leetcode_ui: true
entry_slug: "2024-05-03-165-compare-version-numbers"
---

[165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/description/) medium
[blog post](https://leetcode.com/problems/compare-version-numbers/solutions/5104929/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03052024-165-compare-version-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cnCdE13wUZo)
![2024-05-03_09-21.webp](/assets/leetcode_daily_images/bd912e31.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/591

#### Problem TLDR

Compare version numbers #medium

#### Intuition

We can use two pointers and scan the strings with O(1) memory. More compact and simple code would be by using a `split`.

#### Approach

* `zip` helps to save some lines of code

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or can be O(1)

#### Code

```kotlin

    fun compareVersion(version1: String, version2: String): Int {
        var r1 = version1.split(".").map { it.toInt() }
        var r2 = version2.split(".").map { it.toInt() }
        val pad = List(abs(r1.size - r2.size)) { 0 }
        return (r1 + pad).zip(r2 + pad).firstOrNull { (a, b) -> a != b }
            ?.let { (a, b) -> a.compareTo(b) } ?: 0
    }

```
```rust

    pub fn compare_version(version1: String, version2: String) -> i32 {
        let v1: Vec<_> = version1.split('.').map(|x| x.parse().unwrap()).collect();
        let v2: Vec<_> = version2.split('.').map(|x| x.parse().unwrap()).collect();
        for i in 0..v1.len().max(v2.len()) {
            let a = if i < v1.len() { v1[i] } else { 0 };
            let b = if i < v2.len() { v2[i] } else { 0 };
            if a < b { return -1 }
            if a > b { return 1 }
        }; 0
    }

```

