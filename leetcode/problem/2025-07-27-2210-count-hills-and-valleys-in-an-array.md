---
layout: leetcode-entry
title: "2210. Count Hills and Valleys in an Array"
permalink: "/leetcode/problem/2025-07-27-2210-count-hills-and-valleys-in-an-array/"
leetcode_ui: true
entry_slug: "2025-07-27-2210-count-hills-and-valleys-in-an-array"
---

[2210. Count Hills and Valleys in an Array](https://leetcode.com/problems/count-hills-and-valleys-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/count-hills-and-valleys-in-an-array/solutions/7010578/kotlin-rust-by-samoylenkodmitry-g4zo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27072025-2210-count-hills-and-valleys?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LxREDojIcMc)
![1.webp](/assets/leetcode_daily_images/4130f6fd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1062

#### Problem TLDR

Count hills and valleys #easy

#### Intuition

We have to remove duplications.

Clever ways to look at the problem:
* dedup
* count `slopes` instead: up, down, up, down...
* compare only hills or valleys values

#### Approach

* duplicates are the corner case, failed the same way 1 year ago
* each language opens a new angle of implementation, try many

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 16ms
    fun countHillValley(n: IntArray) =
        (1..<n.lastIndex).count { i ->
            val a = n[0]; val b = n[i]; val c = n[i + 1]
            (a > b && b < c || a < b && b > c).also { if (it) n[0] = b }
        }

```
```kotlin

// 11ms
    fun countHillValley(n: IntArray): Int {
        var p = 0
        return n.filter { x -> x != p.also { p = x } }
                .windowed(3)
                .count { (a, b, c) -> (a > b) == (b < c) }
    }

```
```kotlin

// 1ms
    fun countHillValley(n: IntArray): Int {
        var p = n[0]; var s = 0
        return max(0, n.count { x ->
            val cs = x.compareTo(p)
            cs != 0 && cs != s.also { s = cs; p = x }} - 1)
    }

```
```rust

// 0ms
    pub fn count_hill_valley(mut n: Vec<i32>) -> i32 {
        n.dedup();
        n.windows(3).filter(|w| (w[0] > w[1]) == (w[1] < w[2])).count() as _
    }

```
```rust

// 0ms
    pub fn count_hill_valley(mut n: Vec<i32>) -> i32 {
        n.iter().dedup().tuple_windows::<(_,_,_)>()
        .filter(|(a, b, c)| (a > b) == (b < c)).count() as _
    }

```
```c++

// 0ms
    int countHillValley(vector<int>& n) {
        int r = 0, p = n[0], s = 0;
        for (int x: n) {
            int cs = x > p ? 1 : x < p ? -1 : 0;
            if (cs) r += cs != s, s = cs; p = x;
        } return max(0, r - 1);
    }

```
```python3

// 0ms
    def countHillValley(self, n: List[int]) -> int:
        d = [x for x,_ in groupby(n)]
        return sum((a > b) == (b < c) for a, b, c in zip(d, d[1:], d[2:]))

```

