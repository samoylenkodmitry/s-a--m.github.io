---
layout: leetcode-entry
title: "1504. Count Submatrices With All Ones"
permalink: "/leetcode/problem/2025-08-21-1504-count-submatrices-with-all-ones/"
leetcode_ui: true
entry_slug: "2025-08-21-1504-count-submatrices-with-all-ones"
---

[1504. Count Submatrices With All Ones](https://leetcode.com/problems/count-submatrices-with-all-ones/description) medium
[blog post](https://leetcode.com/problems/count-submatrices-with-all-ones/solutions/7105836/kotlin-rust-by-samoylenkodmitry-m5w8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21082025-1504-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UjUyxBqRjgU)

![1.webp](/assets/leetcode_daily_images/2afa9b96.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1088

#### Problem TLDR

Count islands of 1 #medium #dp

#### Intuition

```j
    //
    //   4
    //   *
    //   *
    // 1 *
    // * x
    // 1 4

    //   3
    //   4   4
    //   * 3 *
    //   * * *
    // 1 * * *
    // * * * x
    // 1 3 3 4
    //
    //   3       5
    //   4   4 4 *
    //   * 3 * * *
    //   * * * * *
    // 1 * * * * *
    // * * * * * x
    // 1 3 3 4 4 5
    //
    //   4       6
    //   5   5 5 *
    //   * 4 * * *
    //   * * * * *
    // 2 * * * * *
    // * * * * * *
    // * * * * * x
    // 2 4 4 5 5 6

    // 1,0,1,1,1,1,1   16
    // 1,1,0,0,0,1,1
    // 2 1       2 2   26
    // 1,1,1,0,0,1,1
    // 3 2 1
    // 1,0,1,0,1,0,1
    // 1,0,1,1,1,0,1
    // 1,1,0,1,1,1,1
    // 1,0,0,1,1,0,1
```
Go row by row, store the heights. For each new position
* it is the bottom right corner of all possible rectangles
* it is equal to the sum of decreasing heights

#### Approach

* reuse the input (not in production or in interview)
* the monotonic stack is not required; idea: it holds only increasing indices, pop while decrease, remove the diff (use the next index in stack or -1)

#### Complexity

- Time complexity:
$$O(n^2m)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 8ms
    fun numSubmat(m: Array<IntArray>): Int {
        var res = 0; val h = IntArray(m[0].size)
        for (r in m) for (x in h.indices) {
            var c = r[x] * ++h[x]; h[x] = c
            for (j in x downTo 0) { c = min(c, h[j]); res += c }
        }
        return res
    }

```
```kotlin

// 53ms
    fun numSubmat(m: Array<IntArray>) =
        m.withIndex().sumOf { (y, h) ->
            val st = Stack<Int>(); var c = 0
            m[y].indices.sumOf { x ->
                h[x] = m[y][x] * (1 + if (y > 0) m[y-1][x] else 0); c += h[x]
                while (st.size > 0 && h[st.peek()] > h[x]) {
                    val j = st.pop()
                    c -= (h[j] - h[x]) * (j - if (st.size > 0) st.peek() else -1)
                }
                st += x; c
            }
        }

```
```rust

// 4ms
    pub fn num_submat(m: Vec<Vec<i32>>) -> i32 {
        let (mut r, mut h) = (0, vec![0; m[0].len()]);
        for y in &m { for x in 0..h.len() {
            let mut c = y[x] * (1 + h[x]); h[x] = c;
            for j in (0..=x).rev() { c = c.min(h[j]); r += c }
        }} r
    }

```
```c++

// 5ms
    int numSubmat(vector<vector<int>>& m) {
        int res = 0; vector<int> h(size(m[0]));
        for (auto& r: m) for (int x = 0; x < size(r); ++x) {
            int c = r[x] * ++h[x]; h[x] = c;
            for (int j = x; j >= 0; --j) res += c = min(c, h[j]);
        } return res;
    }

```
```python

// 772ms
    def numSubmat(_, m):
        r = 0; h = [0] * len(m[0])
        for y in m:
            for i,x in enumerate(y):
                c=x and h[i]+1;h[i]=c
                r+=sum((c:=min(c, h[j])) for j in range(i, -1, -1))
        return r

```

