---
layout: leetcode-entry
title: "3024. Type of Triangle"
permalink: "/leetcode/problem/2025-05-19-3024-type-of-triangle/"
leetcode_ui: true
entry_slug: "2025-05-19-3024-type-of-triangle"
---

[3024. Type of Triangle](https://leetcode.com/problems/type-of-triangle/description) easy
[blog post](https://leetcode.com/problems/type-of-triangle/solutions/6758249/kotlin-rust-by-samoylenkodmitry-r1yj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19052025-3024-type-of-triangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tolXlmCf2rY)
![1.webp](/assets/leetcode_daily_images/4d8c7c9f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/993

#### Problem TLDR

Triangle type by lengths #easy

#### Intuition

Was surprisingly hard to work all the corner cases.

#### Approach

* a = max(), b = min(), c = sum() - a - b

#### Complexity

- Time complexity:
$$O(brain)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun triangleType(n: IntArray) =
        listOf("none", "equilateral", "isosceles", "scalene")[
        if (2 * n.max() >= n.sum()) 0 else n.toSet().size]

```
```rust

    pub fn triangle_type(mut n: Vec<i32>) -> String {
        n.sort();
        (if n[2] >= n[0] + n[1] { "none" } else
        if n[0] == n[2] { "equilateral" } else
        if n[0] == n[1] || n[1] == n[2] { "isosceles" } else { "scalene" }).into()
    }

```
```c++

    string triangleType(vector<int>& nums) {
        int f[101]={}, m = 0, mf = 0, s = 0;
        for (int x: nums) mf = max(mf, ++f[x]), m = max(m, x), s += x;
        return array{"none", "scalene", "isosceles", "equilateral"}[2 * m >= s ? 0 : mf];
    }

```

