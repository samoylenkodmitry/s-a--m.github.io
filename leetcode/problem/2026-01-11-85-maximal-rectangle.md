---
layout: leetcode-entry
title: "85. Maximal Rectangle"
permalink: "/leetcode/problem/2026-01-11-85-maximal-rectangle/"
leetcode_ui: true
entry_slug: "2026-01-11-85-maximal-rectangle"
---

[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/description/) hard
[blog post](https://leetcode.com/problems/maximal-rectangle/solutions/7485928/kotlin-rust-by-samoylenkodmitry-m1iq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11012026-85-maximal-rectangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ct21KpLMtA4)

![c5a04d2d-e0ec-4ee9-8db9-4eed14f73561 (1).webp](/assets/leetcode_daily_images/89e3ace2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1234

#### Problem TLDR

Max all-1 rectangle in 2D matrix #hard #monotonic_stack

#### Intuition

The histogram solution: use monotonic increasing stack for each row. Update result when popping values from it.

#### Approach

* to avoid duplicated code after the row is finished, use sentinel 0 or just one extra iteration
* we can store heights in a separate array or just modify matrix
* for the `left` x coordinate use the value `before` popped value

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 31ms
    fun maximalRectangle(m: Array<CharArray>) = Stack<Int>().run {
        var res = 0; for (y in m.indices) for (x in 0..m[0].size) {
            if (y>0&&x<m[0].size) if (m[y][x]>'0')m[y][x]+=m[y-1][x]-'0'
            while (size > 0 && (x==m[0].size||m[y][peek()]>m[y][x]))
                res = max(res, (m[y][pop()]-'0')*(x-if(size>0)peek()+1 else 0))
            if (x < m[0].size) this += x else clear()
        }; res
    }
```
```rust
// 0ms
    pub fn maximal_rectangle(mut m: Vec<Vec<char>>) -> i32 {
        let (mut q, mut r) = (vec![], 0);
        for y in 0..m.len() { for x in 0..=m[0].len() {
            if y > 0 && x < m[0].len() { if m[y][x] > '0' { m[y][x] = ('1' as u8 + (m[y-1][x] as u8 -b'0'))as char}}
            while q.len() > 0 && (x == m[0].len() || m[y][q[q.len()-1]] > m[y][x]) {
                r = r.max((m[y][q.pop().unwrap()]as u8 -b'0')as i32 * if q.len()>0 {x-q[q.len()-1]-1} else {x}as i32)}
            if x < m[0].len() { q.push(x) } else { q.clear() }
        }} r
    }
```

