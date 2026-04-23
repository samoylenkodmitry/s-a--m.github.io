---
layout: leetcode-entry
title: "498. Diagonal Traverse"
permalink: "/leetcode/problem/2025-08-25-498-diagonal-traverse/"
leetcode_ui: true
entry_slug: "2025-08-25-498-diagonal-traverse"
---

[498. Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/description) medium
[blog post](https://leetcode.com/problems/diagonal-traverse/solutions/7120158/kotlin-rust-by-samoylenkodmitry-705d/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25082025-498-diagonal-traverse?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EzDVtfp31EU)

![1.webp](/assets/leetcode_daily_images/183a36c4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1092

#### Problem TLDR

Matrix diagonal traversal #medium

#### Intuition

Use the fact: diagonal x + y = constant

#### Approach

* use d%2 to check if should go up

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

// 39ms
    fun findDiagonalOrder(m: Array<IntArray>) = buildList {
        for (d in 0..m.lastIndex+m[0].lastIndex)
            for (x in (max(0,d-m.lastIndex)..min(d, m[0].lastIndex))
                .let { if (d%2>0) it.reversed() else it}) this += m[d-x][x]
    }

```

```rust

// 0ms
    pub fn find_diagonal_order(m: Vec<Vec<i32>>) -> Vec<i32> {
        (0..=m.len()+m[0].len()-2).flat_map(|d| {
            let (a, b) = (d.saturating_sub(m.len()-1), d.min(m[0].len()-1));
            let mut n: Vec<_> = (a..=b).map(|x| m[d-x][x]).collect();
            if d%2>0 { n.reverse() }; n
        }).collect()
    }

```
```c++

// 4ms
    vector<int> findDiagonalOrder(vector<vector<int>>& m) {
        vector<int>r; int w = size(m[0]), h = size(m);
        for (int d = 0; d <= w+h-2; ++d) {
            int b = min(d, w-1), a = max(0, d-h+1);
            for (int x = (d%2)*b+(1-d%2)*a; d%2 && x >= a || d%2<1 && x <= b; d%2?--x:++x)
                r.push_back(m[d-x][x]);
        } return r;
    }

```
```python

// 27ms
    def findDiagonalOrder(_, m):
        h,w=len(m),len(m[0]); return [m[d-x][x]
        for d in range(h+w-1)
        for x in range(max(0,d-h+1),min(d,w-1)+1)[::1-2*(d&1)]]

```

