$(document).ready(function () {
    $("a[href*='leetcode.com']").on("click", function (event) {
        event.preventDefault();
        let link = $(this);
        let container = $("<div class='expandable-link-container'></div>");
        link.after(container);

        if (!container.data("loaded")) {
            container.load(link.attr("href") + " .problem-content", function () {
                container.data("loaded", true);
            });
        }

        container.slideToggle("fast");
    });
});
