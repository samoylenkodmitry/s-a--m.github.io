---
layout: leetcode-entry
title: "Maximum 69 Number"
permalink: "/leetcode/problem/2022-11-07-maximum-69-number/"
leetcode_ui: true
entry_slug: "2022-11-07-maximum-69-number"
---

[https://leetcode.com/problems/maximum-69-number/](https://leetcode.com/problems/maximum-69-number/) easy

```kotlin

    fun maximum69Number (num: Int): Int {
        var n = num
        if (6666 <= n && n <= 6999) return num + 3000
        if (n > 9000) n -= 9000
        if (666 <= n && n <= 699) return num + 300
        if (n > 900) n -= 900
        if (66 <= n && n <= 69) return num + 30
        if (n > 90) n -= 90
        if (6 == n) return num + 3
        return num
    }

```

Explanation:
The simplest implementations would be converting to array of digits, replacing the first and converting back.
However we can observe that numbers are in range 6-9999, so we can hardcode some logic.

Speed: O(1), Memory: O(1)

