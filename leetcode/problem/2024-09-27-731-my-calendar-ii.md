---
layout: leetcode-entry
title: "731. My Calendar II"
permalink: "/leetcode/problem/2024-09-27-731-my-calendar-ii/"
leetcode_ui: true
entry_slug: "2024-09-27-731-my-calendar-ii"
---

[731. My Calendar II](https://leetcode.com/problems/my-calendar-ii/description/) medium
[blog post](https://leetcode.com/problems/my-calendar-ii/solutions/5839078/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27092024-731-my-calendar-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2m1yNSj55-s)
![1.webp](/assets/leetcode_daily_images/156e9d88.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/748

#### Problem TLDR

Add intervals intersecting less than two times #medium #line_sweep

#### Intuition

Let's observe the problem:

```j

    // 0123456
    // ---  --  0,3  5,7
    //   ----   2,6            0,3 2,6 5,7
    //  ---     1,4

```

One way to solve the overlapping intervals is a line sweep algorithm: sort intervals, and increase the `counter` on each `start`, decrease on each `end`. This algorithm will take at least O(n) on each call, or O(nlog(n)) for a shorter code with sort instead of binary search.

Another, more clever way, is to maintain a second list of intervals of `intersections`.

#### Approach

* for the line sweep, use `end - 1`, and sort by the `start` and put `ends` after the `starts`

#### Complexity

- Time complexity:
$$O(n)$$, or O(n^2)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class MyCalendarTwo() {
    var list = listOf<Pair<Int, Int>>()
    fun book(start: Int, end: Int): Boolean {
        val se = (list + (start to 1) + ((end - 1) to -1))
            .sortedWith(compareBy({ it.first }, { -it.second }))
        var count = 0
        return if (se.any { (_, c) -> count += c; count > 2 })
            false else { list = se; true }
    }
}

```
```rust

#[derive(Default)] struct MyCalendarTwo((Vec<(i32, i32)>, Vec<(i32, i32)>));
impl MyCalendarTwo {
    fn new() -> Self { Self::default() }
    fn book(&mut self, start: i32, end: i32) -> bool {
        for &(s, e) in &self.0.0 { if start < e && end > s { return false; }}
        for &(s, e) in &self.0.1 { if start < e && end > s {
            self.0.0.push((start.max(s), end.min(e))); }}
        self.0.1.push((start, end)); true
    }
}

```
```c++

class MyCalendarTwo {
public:
    vector<pair<int, int>> booking, overlap;
    bool book(int start, int end) {
        for (const auto& [s, e]: overlap) if (start < e && end > s) return false;
        for (const auto& [s, e]: booking) if (start < e && end > s)
            overlap.emplace_back(max(start, s), min(end, e));
        booking.emplace_back(start, end); return true;
    }
};

```

