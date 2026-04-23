---
layout: leetcode-entry
title: "3000. Maximum Area of Longest Diagonal Rectangle"
permalink: "/leetcode/problem/2025-08-26-3000-maximum-area-of-longest-diagonal-rectangle/"
leetcode_ui: true
entry_slug: "2025-08-26-3000-maximum-area-of-longest-diagonal-rectangle"
---

[3000. Maximum Area of Longest Diagonal Rectangle](https://leetcode.com/problems/maximum-area-of-longest-diagonal-rectangle/description/) medium
[blog post](https://leetcode.com/problems/maximum-area-of-longest-diagonal-rectangle/solutions/7123967/kotlin-rust-by-samoylenkodmitry-1gvi/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26082025-3000-maximum-area-of-longest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/w7tNJWpFVLA)

![1.webp](/assets/leetcode_daily_images/a76e0271.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1093

#### Problem TLDR

Max area of max diagonal rectangles #easy

#### Intuition

Single iteration:
* update max diagonal, forget max area
* update max area

#### Approach

* or we can find a max of a single variable `diagonal+area`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 11ms
    fun areaOfMaxDiagonal(d: Array<IntArray>) = d
        .maxOf { (w,h) -> (w*w+h*h) * 10000 + w*h }%10000

```
```rust

// 0ms
    pub fn area_of_max_diagonal(d: Vec<Vec<i32>>) -> i32 {
        d.iter().map(|d|(d[0]*d[0]+d[1]*d[1],d[0]*d[1])).max().unwrap().1
    }

```
```c++

// 0ms
    int areaOfMaxDiagonal(vector<vector<int>>& d) {
        int x = 0;
        for (auto d: d) x = max(x, (d[0]*d[0]+d[1]*d[1])*10000+d[0]*d[1]);
        return x%10000;
    }

```
```python

// 0ms
    areaOfMaxDiagonal= lambda _,d:max((w*w+h*h,w*h) for w,h in d)[1]

```

