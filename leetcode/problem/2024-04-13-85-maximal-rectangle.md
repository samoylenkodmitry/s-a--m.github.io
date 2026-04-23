---
layout: leetcode-entry
title: "85. Maximal Rectangle"
permalink: "/leetcode/problem/2024-04-13-85-maximal-rectangle/"
leetcode_ui: true
entry_slug: "2024-04-13-85-maximal-rectangle"
---

[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/description/) hard
[blog post](https://leetcode.com/problems/maximal-rectangle/solutions/5015123/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13042024-85-maximal-rectangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2ltM9lpAomQ)
![2024-04-13_09-13.webp](/assets/leetcode_daily_images/a1555081.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/570

#### Problem TLDR

Max `1`-only area in a `0-1` matrix #hard

#### Intuition

The n^4 solution is kind of trivial, just precompute the prefix sums, then do some geometry:
![2024-04-13_09-101.webp](/assets/leetcode_daily_images/61ad09cf.webp)

The trick here is to observe a subproblem (https://leetcode.com/problems/largest-rectangle-in-histogram/):
![2024-04-13_09-102.webp](/assets/leetcode_daily_images/3a97d3fe.webp)
This can be solved using a `Monotonic Increasing Stack` technique:

```j
    //i0 1 2 3 4
    // 3 1 3 2 2
    //0*          3
    //1  *        1
    //2    *      1 3
    //3      *    1 3 2  -> 1 2
    //4        *  1 2 2
    //           * empty
```
Pop all positions smaller than the current heights. Careful with the area calculation though, the height will be the popping one, and the width is a distance between popped and a new top.

#### Approach

There are some tricks:
* using a sentinel 0-height at the end of `h` will help to save some lines of code
* Stack object can be reused

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin []

    fun maximalRectangle(matrix: Array<CharArray>): Int = with(Stack<Int>()) {
        val h = IntArray(matrix[0].size + 1)
        var max = 0
        for (y in matrix.indices) for (x in h.indices) {
            if (x < h.size - 1) h[x] = if (matrix[y][x] > '0') 1 + h[x] else 0
            while (size > 0 && h[peek()] > h[x])
                max = max(max, h[pop()] * if (size > 0) x - peek() - 1 else x)
            if (x < h.size - 1) push(x) else clear()
        }
        max
    }

```
```rust

    pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
        let (mut st, mut h, mut max) = (vec![], vec![0; matrix[0].len() + 1], 0);
        for y in 0..matrix.len() {
            for x in 0..h.len() {
                if x < h.len() - 1 { h[x] = if matrix[y][x] > '0' { 1 + h[x] } else { 0 }}
                while st.len() > 0 && h[*st.last().unwrap()] > h[x] {
                    let l = st.pop().unwrap();
                    max = max.max(h[l] * if st.len() > 0 { x - *st.last().unwrap() - 1 } else { x })
                }
                if x < h.len() - 1 { st.push(x) } else { st.clear() }
            }
        }
        max as i32
    }

```

